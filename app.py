
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
    TextIteratorStreamer
)

# Configuration Constants
MODEL_ID= "FuseAI/FuseO1-DeepSeekR1-QwQ-SkyT1-32B-Preview"


# Understand]: Analyze the question to identify key details and clarify the goal.
# [Plan]: Outline a logical, step-by-step approach to address the question or problem.
# [Reason]: Execute the plan, applying logical reasoning, calculations, or analysis to reach a conclusion. Document each step clearly.
# [Reflect]: Review the reasoning and the final answer to ensure it is accurate, complete, and adheres to the principle of openness.
# [Respond]: Present a well-structured and transparent answer, enriched with supporting details as needed.
# Use these tags as headers in your response to make your thought process easy to follow and aligned with the principle of openness.

DEFAULT_SYSTEM_PROMPT ="""
You are a reasoning assistant specialized in problem-solving, You should think Step by Step.
Reasoning: {{reasoning}}
Answer: {{answer}}
"""
# UI Configuration
TITLE = "<h1><center>AI Reasoning Assistant</center></h1>"
PLACEHOLDER = "Ask me anything! I'll think through it step by step."

CSS = """
.duplicate-button {
    margin: auto !important;
    color: white !important;
    background: black !important;
    border-radius: 100vh !important;
}
h3 {
    text-align: center;
}
.message-wrap {
    overflow-x: auto;
}
.message-wrap p {
    margin-bottom: 1em;
}
.message-wrap pre {
    background-color: #f6f8fa;
    border-radius: 3px;
    padding: 16px;
    overflow-x: auto;
}
.message-wrap code {
    background-color: rgba(175,184,193,0.2);
    border-radius: 3px;
    padding: 0.2em 0.4em;
    font-family: monospace;
}
.custom-tag {
    color: #0066cc;
    font-weight: bold;
}
.chat-area {
    height: 500px !important;
    overflow-y: auto !important;
}
"""

def initialize_model():
    """Initialize the model with appropriate configurations"""
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
        bnb_8bit_quant_type="nf4",
        bnb_8bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID , trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
        # attn_implementation="flash_attention_2",
        trust_remote_code=True,
        quantization_config=quantization_config

    )

    return model, tokenizer

def format_text(text):
    """Format text with proper spacing and tag highlighting (but keep tags visible)"""
    tag_patterns = [
        (r'<Thinking>', '\n<Thinking>\n'),
        (r'</Thinking>', '\n</Thinking>\n'),
        (r'<Critique>', '\n<Critique>\n'),
        (r'</Critique>', '\n</Critique>\n'),
        (r'<Revising>', '\n<Revising>\n'),
        (r'</Revising>', '\n</Revising>\n'),
        (r'<Final>', '\n<Final>\n'),
        (r'</Final>', '\n</Final>\n')
    ]
    
    formatted = text
    for pattern, replacement in tag_patterns:
        formatted = re.sub(pattern, replacement, formatted)
    
    formatted = '\n'.join(line for line in formatted.split('\n') if line.strip())
    
    return formatted

def format_chat_history(history):
    """Format chat history for display, keeping tags visible"""
    formatted = []
    for user_msg, assistant_msg in history:
        formatted.append(f"User: {user_msg}")
        if assistant_msg:
            formatted.append(f"Assistant: {assistant_msg}")
    return "\n\n".join(formatted)
    
def create_examples():
    """Create example queries for the UI"""
    return [
        "Explain the concept of artificial intelligence.",
        "How does photosynthesis work?",
        "What are the main causes of climate change?",
        "Describe the process of protein synthesis.",
        "What are the key features of a democratic government?",
        "Explain the theory of relativity.",
        "How do vaccines work to prevent diseases?",
        "What are the major events of World War II?",
        "Describe the structure of a human cell.",
        "What is the role of DNA in genetics?"
    ]

@spaces.GPU(duration=660)
def chat_response(
    message: str,
    history: list,
    chat_display: str,
    system_prompt: str,
    temperature: float = 0.3,
    max_new_tokens: int =4096 ,
    top_p: float = 0.1,
    top_k: int = 45,
    penalty: float = 1.5,
):
    """Generate chat responses, keeping tags visible in the output"""
    conversation = [
        {"role": "system", "content": system_prompt}
    ]
    
    for prompt, answer in history:
        conversation.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}
        ])
    
    conversation.append({"role": "user", "content": message})
    
    input_ids = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    streamer = TextIteratorStreamer(
        tokenizer,
        timeout=60.0,
        skip_prompt=True,
        skip_special_tokens=True
    )
    
    generate_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False if temperature == 0 else True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        repetition_penalty=penalty,
        streamer=streamer,
    )
    
    buffer = ""
    
    with torch.no_grad():
        thread = Thread(target=model.generate, kwargs=generate_kwargs)
        thread.start()
        
        history = history + [[message, ""]]
        
        for new_text in streamer:
            buffer += new_text
            formatted_buffer = format_text(buffer)
            history[-1][1] = formatted_buffer
            chat_display = format_chat_history(history)
            
            yield history, chat_display

def process_example(example: str) -> tuple:
    """Process example query and return empty history and updated display"""
    return [], f"User: {example}\n\n"

def main():
    """Main function to set up and launch the Gradio interface"""
    global model, tokenizer
    model, tokenizer = initialize_model()
    
    with gr.Blocks(css=CSS, theme="soft") as demo:
        gr.HTML(TITLE)
        gr.DuplicateButton(
            value="Duplicate Space for private use",
            elem_classes="duplicate-button"
        )
        
        with gr.Row():
            with gr.Column():
                chat_history = gr.State([])
                chat_display = gr.TextArea(
                    value="",
                    label="Chat History",
                    interactive=False,
                    elem_classes=["chat-area"],
                )
                
                message = gr.TextArea(
                    placeholder=PLACEHOLDER,
                    label="Your message",
                    lines=3
                )
                
                with gr.Row():
                    submit = gr.Button("Send")
                    clear = gr.Button("Clear")
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    system_prompt = gr.TextArea(
                        value=DEFAULT_SYSTEM_PROMPT,
                        label="System Prompt",
                        lines=5,
                    )
                    temperature = gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=0.1,
                        value=0.3,
                        label="Temperature",
                    )
                    max_tokens = gr.Slider(
                        minimum=128,
                        maximum=32000,
                        step=128,
                        value=4096,
                        label="Max Tokens",
                    )
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        step=0.1,
                        value=0.8,
                        label="Top-p",
                    )
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=45,
                        label="Top-k",
                    )
                    penalty = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        step=0.1,
                        value=1.5,
                        label="Repetition Penalty",
                    )
                
                examples = gr.Examples(
                    examples=create_examples(),
                    inputs=[message],
                    outputs=[chat_history, chat_display],
                    fn=process_example,
                    cache_examples=False,
                )
        
        # Set up event handlers
        submit_click = submit.click(
            chat_response,
            inputs=[
                message,
                chat_history,
                chat_display,
                system_prompt,
                temperature,
                max_tokens,
                top_p,
                top_k,
                penalty,
            ],
            outputs=[chat_history, chat_display],
            show_progress=True,
        )
        
        message.submit(
            chat_response,
            inputs=[
                message,
                chat_history,
                chat_display,
                system_prompt,
                temperature,
                max_tokens,
                top_p,
                top_k,
                penalty,
            ],
            outputs=[chat_history, chat_display],
            show_progress=True,
        )
        
        clear.click(
            lambda: ([], ""),
            outputs=[chat_history, chat_display],
            show_progress=True,
        )
        
        submit_click.then(lambda: "", outputs=message)
        message.submit(lambda: "", outputs=message)
    
    return demo

if __name__ == "__main__":
    demo = main()
    demo.launch()