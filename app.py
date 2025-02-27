import torch
import numpy as np
import gradio as gr
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
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    
    return messages

def generate_response_with_visualization(messages, gen_length=64, steps=32, constraints=None):
    """
    Generate text with LLaDA model with visualization of the denoising process
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        
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
    
    # Initialize visualization states for just the response part
    visualization_states = []
    
    # Add initial state (all masked) - only for the response part
    initial_state = [(MASK_TOKEN, "#444444") for _ in range(gen_length)]
    visualization_states.append(initial_state)
    
    # Apply constraints to the initial state
    for pos, token_id in processed_constraints.items():
        absolute_pos = prompt_length + pos
        if absolute_pos < x.shape[1]:
            x[:, absolute_pos] = token_id
    
    # Calculate timesteps
    timesteps = torch.linspace(1.0, 0.0, steps + 1)[:-1]
    
    # Keep track of already revealed tokens
    revealed_tokens = torch.zeros(1, gen_length, dtype=torch.bool).to(device)
    
    for step, t in enumerate(timesteps):
        # Current t to next t
        s = t - 1.0 / steps if step < steps - 1 else 0
        
        # Get all mask positions in the current sequence
        mask_indices = (x == MASK_ID)
        
        # Skip if no masks
        if not mask_indices.any():
            break
            
        # Get logits from the model
        logits = model(x).logits
        
        # Get the top predictions
        x0 = torch.argmax(logits, dim=-1)
        
        # Get probabilities for visualization
        probs = torch.softmax(logits, dim=-1)
        top_probs = torch.max(probs, dim=-1)[0]
        
        # Apply the predictions where we have masks
        x_old = x.clone()
        x = torch.where(mask_indices, x0, x)
        
        # Calculate how many tokens should remain masked at next step
        total_len = gen_length
        current_t_value = float(t)
        next_t_value = float(s)
        
        # Linear schedule: t=1 → all masked, t=0 → none masked
        current_masks_expected = int(current_t_value * total_len)
        next_masks_expected = int(next_t_value * total_len)
        
        # How many to unmask in this step
        tokens_to_unmask = current_masks_expected - next_masks_expected
        
        if tokens_to_unmask > 0 and mask_indices.any():
            # Get confidence scores for currently masked tokens
            confidence_scores = top_probs[mask_indices]
            
            # Sort confidence scores
            sorted_indices = torch.argsort(confidence_scores, descending=True)
            
            # Select which tokens to keep masked (the lowest confidence ones)
            indices_to_remask = sorted_indices[tokens_to_unmask:]
            
            # Get the actual indices in the sequence
            mask_positions = torch.where(mask_indices)[1]
            positions_to_remask = mask_positions[indices_to_remask]
            
            # Remask these positions
            x[:, positions_to_remask] = MASK_ID
        
        # Ensure constraints are maintained
        for pos, token_id in processed_constraints.items():
            absolute_pos = prompt_length + pos
            if absolute_pos < x.shape[1]:
                x[:, absolute_pos] = token_id
        
        # Create visualization state ONLY for the response part
        current_state = []
        
        # Update which tokens are newly revealed in this step
        for i in range(gen_length):
            pos = prompt_length + i  # Absolute position in the sequence
            
            if x[0, pos] == MASK_ID:
                # Still masked
                current_state.append((MASK_TOKEN, "#444444"))  # Dark gray for masks
                
            elif x_old[0, pos] == MASK_ID:
                # Newly revealed in this step
                token = tokenizer.decode([x[0, pos].item()], skip_special_tokens=True)
                confidence = float(top_probs[0, pos].cpu())
                
                # Color based on confidence: red (low) to green (high)
                if confidence < 0.3:
                    color = "#FF6666"  # Light red
                elif confidence < 0.7:
                    color = "#FFAA33"  # Orange
                else:
                    color = "#66CC66"  # Light green
                    
                current_state.append((token, color))
                revealed_tokens[0, i] = True
                
            else:
                # Previously revealed
                token = tokenizer.decode([x[0, pos].item()], skip_special_tokens=True)
                current_state.append((token, "#6699CC"))  # Light blue
        
        visualization_states.append(current_state)
    
    # Extract final text (just the assistant's response)
    response_tokens = x[0, prompt_length:]
    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    # Clean the response text
    final_text = tokenizer.decode(response_tokens, 
                                 skip_special_tokens=True,
                                 clean_up_tokenization_spaces=True)
    
    return visualization_states, final_text

css = '''
.category-legend{display:none}
button{height: 60px}
'''
with gr.Blocks(css=css) as demo:
    gr.Markdown("# LLaDA - Large Language Diffusion Model demo")
    gr.Markdown("[model](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct), [project page](https://ml-gsai.github.io/LLaDA-demo/)")
    
    # STATE MANAGEMENT - IMPORTANT
    # We use a dedicated state to track the full conversation history
    chat_history = gr.State([])
    
    # UI COMPONENTS
    # Chatbot for displaying messages
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
    # Visualization and response components
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
        
        
        visualization_delay = gr.Slider(
            minimum=0.0, maximum=1.0, value=0.1, step=0.1, visible=False,
            label="Visualization Delay (seconds)"
        )
    
    # Current response text box
    current_response = gr.Textbox(
        label="Current Response",
        placeholder="The assistant's response will appear here...",
        lines=3,
        visible=False
    )
    
    # Clear button
    clear_btn = gr.Button("Clear Conversation")
    
    # Example inputs
    gr.Examples(
        [
            ["Tell me a short joke", 64, 32, ""],
            ["Write a short story", 64, 32, "0:Once, 5:upon, 10:time"],
            ["Explain quantum computing", 64, 32, ""],
        ],
        [user_input, gen_length, steps, constraints_input],
    )
    
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
        
    def bot_response(history, gen_length, steps, constraints, delay):
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
                messages, 
                gen_length=gen_length, 
                steps=steps,
                constraints=parsed_constraints
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
        inputs=[chat_history, gen_length, steps, constraints_input, visualization_delay],
        outputs=[chatbot_ui, output_vis, current_response]
    )
    
    send_click.then(
        fn=bot_response,
        inputs=[chat_history, gen_length, steps, constraints_input, visualization_delay],
        outputs=[chatbot_ui, output_vis, current_response]
    )
    
return demo

# Launch the demo
if __name__ == "__main__":
    demo.queue().launch(share=True)
