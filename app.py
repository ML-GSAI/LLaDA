import torch
import numpy as np
import gradio as gr
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import time
import re

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, 
                                  torch_dtype=torch.bfloat16).to(device)

# Constants
MASK_TOKEN = "[MASK]"
MASK_ID = 126336  # The token ID of [MASK] in LLaDA

def parse_constraints(constraints_text):
    """Parse constraints in format: 'position:word, position:word, ...'"""
    constraints = {}
    if not constraints_text:
        return constraints
        
    parts = constraints_text.split(',')
    for part in parts:
        if ':' not in part:
            continue
        pos_str, word = part.split(':', 1)
        try:
            pos = int(pos_str.strip())
            word = word.strip()
            if word and pos >= 0:
                constraints[pos] = word
        except ValueError:
            continue
    
    return constraints

def format_chat_history(history):
    """
    Format chat history for the LLaDA model
    
    Args:
        history: List of [user_message, assistant_message] pairs
        
    Returns:
        Formatted conversation for the model
    """
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:  # Skip if None (for the latest user message)
            messages.append({"role": "assistant", "content": assistant_msg})
    
    return messages

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature <= 0:
        return logits
        
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def generate_response_with_visualization(model, tokenizer, device, messages, gen_length=64, steps=32, 
                                         constraints=None, temperature=0.0, cfg_scale=0.0, block_length=32,
                                         remasking='low_confidence'):
    """
    Generate text with LLaDA model with visualization using the same sampling as in generate.py
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        gen_length: Length of text to generate
        steps: Number of denoising steps
        constraints: Dictionary mapping positions to words
        temperature: Sampling temperature
        cfg_scale: Classifier-free guidance scale
        block_length: Block length for semi-autoregressive generation
        remasking: Remasking strategy ('low_confidence' or 'random')
        
    Returns:
        List of visualization states showing the progression and final text
    """
    
    # Process constraints
    if constraints is None:
        constraints = {}
        
    # Convert any string constraints to token IDs
    processed_constraints = {}
    for pos, word in constraints.items():
        tokens = tokenizer.encode(" " + word, add_special_tokens=False)
        for i, token_id in enumerate(tokens):
            processed_constraints[pos + i] = token_id
    
    # Prepare the prompt using chat template
    chat_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(chat_input)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    
    # For generation
    prompt_length = input_ids.shape[1]
    
    # Initialize the sequence with masks for the response part
    x = torch.full((1, prompt_length + gen_length), MASK_ID, dtype=torch.long).to(device)
    x[:, :prompt_length] = input_ids.clone()
    
    # Initialize visualization states for the response part
    visualization_states = []
    
    # Add initial state (all masked)
    initial_state = [(MASK_TOKEN, "#444444") for _ in range(gen_length)]
    visualization_states.append(initial_state)
    
    # Apply constraints to the initial state
    for pos, token_id in processed_constraints.items():
        absolute_pos = prompt_length + pos
        if absolute_pos < x.shape[1]:
            x[:, absolute_pos] = token_id
    
    # Mark prompt positions to exclude them from masking during classifier-free guidance
    prompt_index = (x != MASK_ID)
    
    # Ensure block_length is valid
    if block_length > gen_length:
        block_length = gen_length
    
    # Calculate number of blocks
    num_blocks = gen_length // block_length
    if gen_length % block_length != 0:
        num_blocks += 1
    
    # Adjust steps per block
    steps_per_block = steps // num_blocks
    if steps_per_block < 1:
        steps_per_block = 1
    
    # Track the current state of x for visualization
    current_x = x.clone()

    # Process each block
    for num_block in range(num_blocks):
        # Calculate the start and end indices for the current block
        block_start = prompt_length + num_block * block_length
        block_end = min(prompt_length + (num_block + 1) * block_length, x.shape[1])
        
        # Get mask indices for the current block
        block_mask_index = (x[:, block_start:block_end] == MASK_ID)
        
        # Skip if no masks in this block
        if not block_mask_index.any():
            continue
        
        # Calculate number of tokens to unmask at each step
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        # Process each step
        for i in range(steps_per_block):
            # Get all mask positions in the current sequence
            mask_index = (x == MASK_ID)
            
            # Skip if no masks
            if not mask_index.any():
                break
            
            # Apply classifier-free guidance if enabled
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = MASK_ID
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits
            
            # Apply Gumbel noise for sampling
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            # Calculate confidence scores for remasking
            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(f"Remasking strategy '{remasking}' not implemented")
            
            # Don't consider positions beyond the current block
            x0_p[:, block_end:] = -float('inf')
            
            # Apply predictions where we have masks
            old_x = x.clone()
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -float('inf'))
            
            # Select tokens to unmask based on confidence
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                # Only consider positions within the current block for unmasking
                block_confidence = confidence[j, block_start:block_end]
                if i < steps_per_block - 1:  # Not the last step
                    # Take top-k confidences
                    _, select_indices = torch.topk(block_confidence, 
                                                  k=min(num_transfer_tokens[j, i].item(), 
                                                       block_confidence.numel()))
                    # Adjust indices to global positions
                    select_indices = select_indices + block_start
                    transfer_index[j, select_indices] = True
                else:  # Last step - unmask everything remaining
                    transfer_index[j, block_start:block_end] = mask_index[j, block_start:block_end]
            
            # Apply the selected tokens
            x = torch.where(transfer_index, x0, x)
            
            # Ensure constraints are maintained
            for pos, token_id in processed_constraints.items():
                absolute_pos = prompt_length + pos
                if absolute_pos < x.shape[1]:
                    x[:, absolute_pos] = token_id
            
            # Create visualization state only for the response part
            current_state = []
            for i in range(gen_length):
                pos = prompt_length + i  # Absolute position in the sequence
                
                if x[0, pos] == MASK_ID:
                    # Still masked
                    current_state.append((MASK_TOKEN, "#444444"))  # Dark gray for masks
                    
                elif old_x[0, pos] == MASK_ID:
                    # Newly revealed in this step
                    token = tokenizer.decode([x[0, pos].item()], skip_special_tokens=True)
                    # Color based on confidence
                    confidence = float(x0_p[0, pos].cpu())
                    if confidence < 0.3:
                        color = "#FF6666"  # Light red
                    elif confidence < 0.7:
                        color = "#FFAA33"  # Orange
                    else:
                        color = "#66CC66"  # Light green
                        
                    current_state.append((token, color))
                    
                else:
                    # Previously revealed
                    token = tokenizer.decode([x[0, pos].item()], skip_special_tokens=True)
                    current_state.append((token, "#6699CC"))  # Light blue
            
            visualization_states.append(current_state)
    
    # Extract final text (just the assistant's response)
    response_tokens = x[0, prompt_length:]
    final_text = tokenizer.decode(response_tokens, 
                               skip_special_tokens=True,
                               clean_up_tokenization_spaces=True)
    
    return visualization_states, final_text

css = '''
.category-legend{display:none}
button{height: 60px}
'''
def create_chatbot_demo():
    with gr.Blocks(css=css) as demo:
        gr.Markdown("# LLaDA - Large Language Diffusion Model Demo")
        gr.Markdown("[model](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct), [project page](https://ml-gsai.github.io/LLaDA-demo/)")
        
        # STATE MANAGEMENT
        chat_history = gr.State([])
        
        # UI COMPONENTS
        with gr.Row():
            with gr.Column(scale=3):
                chatbot_ui = gr.Chatbot(label="Conversation", height=500)
                
                # Message input
                with gr.Group():
                    with gr.Row():
                        user_input = gr.Textbox(
                            label="Your Message", 
                            placeholder="Type your message here...",
                            show_label=False
                        )
                        send_btn = gr.Button("Send")
                
                constraints_input = gr.Textbox(
                    label="Word Constraints", 
                    info="This model allows for placing specific words at specific positions using 'position:word' format. Example: 1st word once, 6th word 'upon' and 11th word 'time', would be: '0:Once, 5:upon, 10:time",
                    placeholder="0:Once, 5:upon, 10:time",
                    value=""
                )
            with gr.Column(scale=2):
                output_vis = gr.HighlightedText(
                    label="Denoising Process Visualization",
                    combine_adjacent=False,
                    show_legend=True,
                )
        
        # Advanced generation settings
        with gr.Accordion("Generation Settings", open=False):
            with gr.Row():
                gen_length = gr.Slider(
                    minimum=16, maximum=128, value=64, step=8,
                    label="Generation Length"
                )
                steps = gr.Slider(
                    minimum=8, maximum=64, value=32, step=4,
                    label="Denoising Steps"
                )
            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.0, step=0.1,
                    label="Temperature"
                )
                cfg_scale = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.0, step=0.1,
                    label="CFG Scale"
                )
            with gr.Row():
                block_length = gr.Slider(
                    minimum=8, maximum=128, value=32, step=8,
                    label="Block Length"
                )
                remasking_strategy = gr.Radio(
                    choices=["low_confidence", "random"],
                    value="low_confidence",
                    label="Remasking Strategy"
                )
            with gr.Row():
                visualization_delay = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.1, step=0.1,
                    label="Visualization Delay (seconds)"
                )
        
        # Current response text box (hidden)
        current_response = gr.Textbox(
            label="Current Response",
            placeholder="The assistant's response will appear here...",
            lines=3,
            visible=False
        )
        
        # Clear button
        clear_btn = gr.Button("Clear Conversation")
        
        # HELPER FUNCTIONS
        def add_message(history, message, response):
            """Add a message pair to the history and return the updated history"""
            history = history.copy()
            history.append([message, response])
            return history
            
        def user_message_submitted(message, history, gen_length, steps, constraints, delay):
            """Process a submitted user message"""
            # Skip empty messages
            if not message.strip():
                # Return current state unchanged
                history_for_display = history.copy()
                return history, history_for_display, "", [], ""
                
            # Add user message to history
            history = add_message(history, message, None)
            
            # Format for display - temporarily show user message with empty response
            history_for_display = history.copy()
            
            # Clear the input
            message_out = ""
            
            # Return immediately to update UI with user message
            return history, history_for_display, message_out, [], ""
            
        def bot_response(history, gen_length, steps, constraints, delay, temperature, cfg_scale, block_length, remasking):
            """Generate bot response for the latest message"""
            if not history:
                return history, [], ""
                
            # Get the last user message
            last_user_message = history[-1][0]
            
            try:
                # Format all messages except the last one (which has no response yet)
                messages = format_chat_history(history[:-1])
                
                # Add the last user message
                messages.append({"role": "user", "content": last_user_message})
                
                # Parse constraints
                parsed_constraints = parse_constraints(constraints)
                
                # Generate response with visualization
                vis_states, response_text = generate_response_with_visualization(
                    model, tokenizer, device, 
                    messages, 
                    gen_length=gen_length, 
                    steps=steps,
                    constraints=parsed_constraints,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    block_length=block_length,
                    remasking=remasking
                )
                
                # Update history with the assistant's response
                history[-1][1] = response_text
                
                # Return the initial state immediately
                yield history, vis_states[0], response_text
                
                # Then animate through visualization states
                for state in vis_states[1:]:
                    time.sleep(delay)
                    yield history, state, response_text
                    
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                print(error_msg)
                
                # Show error in visualization
                error_vis = [(error_msg, "red")]
                
                # Don't update history with error
                yield history, error_vis, error_msg
        
        def clear_conversation():
            """Clear the conversation history"""
            return [], [], "", []
        
        # EVENT HANDLERS
        
        # Clear button handler
        clear_btn.click(
            fn=clear_conversation,
            inputs=[],
            outputs=[chat_history, chatbot_ui, current_response, output_vis]
        )
        
        # User message submission flow (2-step process)
        # Step 1: Add user message to history and update UI
        msg_submit = user_input.submit(
            fn=user_message_submitted,
            inputs=[user_input, chat_history, gen_length, steps, constraints_input, visualization_delay],
            outputs=[chat_history, chatbot_ui, user_input, output_vis, current_response]
        )
        
        # Also connect the send button
        send_click = send_btn.click(
            fn=user_message_submitted,
            inputs=[user_input, chat_history, gen_length, steps, constraints_input, visualization_delay],
            outputs=[chat_history, chatbot_ui, user_input, output_vis, current_response]
        )
        
        # Step 2: Generate bot response
        # This happens after the user message is displayed
        msg_submit.then(
            fn=bot_response,
            inputs=[
                chat_history, gen_length, steps, constraints_input, 
                visualization_delay, temperature, cfg_scale, block_length,
                remasking_strategy
            ],
            outputs=[chatbot_ui, output_vis, current_response]
        )
        
        send_click.then(
            fn=bot_response,
            inputs=[
                chat_history, gen_length, steps, constraints_input, 
                visualization_delay, temperature, cfg_scale, block_length,
                remasking_strategy
            ],
            outputs=[chatbot_ui, output_vis, current_response]
        )
        
    return demo

# Launch the demo
if __name__ == "__main__":
    demo = create_chatbot_demo()
    demo.queue().launch(share=True)
