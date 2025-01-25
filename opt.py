import os
import re
import time
import torch
import spaces
import gradio as gr
from threading import Thread
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList
)

# Configuration Constants
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

# Enhanced System Prompt
DEFAULT_SYSTEM_PROMPT = """You are an Expert Reasoning Assistant. Follow these steps:
[Understand]: Analyze key elements and clarify objectives
[Plan]: Outline step-by-step methodology
[Reason]: Execute plan with detailed analysis
[Verify]: Check logic and evidence
[Conclude]: Present structured conclusion

Use these section headers and maintain technical accuracy with clear explanations."""

# UI Configuration
TITLE = """
<h1 align="center" style="color: #2d3436; margin-bottom: 0">🧠 AI Reasoning Assistant</h1>
<p align="center" style="color: #636e72; margin-top: 0">DeepSeek-R1-Distill-Qwen-14B</p>
"""
CSS = """
.gr-chatbot { min-height: 500px !important; border-radius: 15px !important; }
.message-wrap pre { background: #f8f9fa !important; padding: 15px !important; }
.thinking-tag { color: #2ecc71; font-weight: 600; }
.plan-tag { color: #e67e22; font-weight: 600; }
.conclude-tag { color: #3498db; font-weight: 600; }
.control-panel { background: #f8f9fa !important; padding: 20px !important; }
footer { visibility: hidden !important; }
"""

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [0]  # Add custom stop tokens here
        return input_ids[0][-1] in stop_ids

def initialize_model():
    """Initialize model with safety checks"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this application")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    return model, tokenizer

def format_response(text):
    """Enhanced formatting with syntax highlighting for reasoning steps"""
    formatted = text.replace("[Understand]", '\n<strong class="thinking-tag">[Understand]</strong>\n')
    formatted = formatted.replace("[Plan]", '\n<strong class="plan-tag">[Plan]</strong>\n')
    formatted = formatted.replace("[Conclude]", '\n<strong class="conclude-tag">[Conclude]</strong>\n')
    return formatted

@spaces.GPU(duration=120)
def chat_response(
    message: str,
    history: list,
    system_prompt: str,
    temperature: float = 0.3,
    max_new_tokens: int = 2048,
    top_p: float = 0.9,
    top_k: int = 50,
    penalty: float = 1.2,
):
    """Improved streaming generator with error handling"""
    try:
        conversation = [{"role": "system", "content": system_prompt}]
        for user, assistant in history:
            conversation.extend([
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant}
            ])
        conversation.append({"role": "user", "content": message})

        input_ids = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        streamer = TextIteratorStreamer(
            tokenizer,
            timeout=30,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generate_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=penalty,
            streamer=streamer,
            stopping_criteria=StoppingCriteriaList([StopOnTokens()])
        )

        buffer = []
        thread = Thread(target=model.generate, kwargs=generate_kwargs)
        thread.start()

        for new_text in streamer:
            buffer.append(new_text)
            partial_result = "".join(buffer)
            
            # Check for complete sections
            if any(tag in partial_result for tag in ["[Understand]", "[Plan]", "[Conclude]"]):
                yield format_response(partial_result)
            else:
                yield format_response(partial_result + " ▌")

        # Final formatting pass
        yield format_response("".join(buffer))

    except Exception as e:
        yield f"⚠️ Error generating response: {str(e)}"

def create_examples():
    """Enhanced examples with diverse use cases"""
    return [
        ["Explain quantum entanglement in simple terms"],
        ["Design a study plan for learning machine learning"],
        ["Compare blockchain and traditional databases"],
        ["How would you optimize AWS costs for a startup?"],
        ["Explain the ethical implications of CRISPR technology"]
    ]

def main():
    """Improved UI layout and interactions"""
    global model, tokenizer
    model, tokenizer = initialize_model()

    with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
        gr.HTML(TITLE)
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    bubble_full_width=False,
                    show_copy_button=True,
                    render=False
                )
                msg = gr.Textbox(
                    placeholder="Enter your question...",
                    label="Ask the Expert",
                    container=False
                )
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear", variant="secondary")

            with gr.Column(scale=1, elem_classes="control-panel"):
                gr.Examples(
                    examples=create_examples(),
                    inputs=msg,
                    label="Example Queries",
                    examples_per_page=5
                )
                
                with gr.Accordion("⚙️ Generation Parameters", open=False):
                    system_prompt = gr.TextArea(
                        value=DEFAULT_SYSTEM_PROMPT,
                        label="System Instructions",
                        lines=5
                    )
                    temperature = gr.Slider(0, 2, value=0.7, label="Creativity")
                    max_tokens = gr.Slider(128, 4096, value=2048, step=128, label="Max Tokens")
                    top_p = gr.Slider(0, 1, value=0.9, step=0.05, label="Focus (Top-p)")
                    penalty = gr.Slider(1, 2, value=1.2, step=0.1, label="Repetition Control")

        # Event handling
        msg.submit(
            chat_response,
            [msg, chatbot, system_prompt, temperature, max_tokens, top_p, penalty],
            [msg, chatbot],
            show_progress="hidden"
        ).then(lambda: "", None, msg)

        submit_btn.click(
            chat_response,
            [msg, chatbot, system_prompt, temperature, max_tokens, top_p, penalty],
            [msg, chatbot],
            show_progress="hidden"
        ).then(lambda: "", None, msg)

        clear_btn.click(lambda: None, None, chatbot, queue=False)

    return demo

if __name__ == "__main__":
    demo = main()
    demo.queue(max_size=20).launch()