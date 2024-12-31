import gradio as gr
import json
from mistralAPI_CHI import process_directory, MistralAPIWrapper
import os

# Define default config values
default_config = {
    "dir_path": "",
    "prefix": "",
    "api_key": "",
    "selected_chi": [],
    "drop_rate_CHI": 0.1
}

# Ensure config.json exists with default values
if not os.path.exists("config.json"):
    with open("config.json", "w") as f:
        json.dump(default_config, f, indent=4)

# Read config from config.json
try:
    with open("config.json", "r") as f:
        config = json.load(f)
except json.JSONDecodeError:
    config = default_config
    
    
def save_config(dir_path, prefix, api_key, selected_chi, drop_rate_CHI):
    config = {
        "dir_path": dir_path,
        "prefix": prefix,
        "api_key": api_key,
        "selected_chi": selected_chi,
        "drop_rate_CHI": drop_rate_CHI
    }
    try:
        with open("config.json", "w") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"Error saving config: {e}")

# Read CHI types from CHI.json
with open("CHI.json", "r", encoding="utf-8") as f:
    chi_json = json.load(f)
# Exclude "CHI_SUBJECT" and "CHI_SUMMARY" from options
chi_options = [key for key in chi_json.keys() if key not in ("CHI_SUBJECT", "CHI_SUMMARY")]

# Define custom CSS styles
css = """
.container {
}
.header {
    text-align: center;
    margin-bottom: 20px;
}
.title {
    font-size: 32px;
    color: #333;
}
.repo-link {
    text-align: center;
    margin-top: 10px;
}
.input-box {
}
.output-box {
}
"""

def caption_directory(dir_path, prefix, api_key, selected_chi, drop_rate_CHI):
    try:
        model = MistralAPIWrapper(api_key=api_key)
    except Exception as e:
        return f"Error initializing model: {e}"
    if not os.path.isdir(dir_path):
        return "Invalid directory path."
    # Determine skip list: all optional CHI types not selected
    selected_chi = selected_chi or []
    optional_chi = [chi for chi in chi_options if chi not in selected_chi]
    skip_CHI = optional_chi
    result = process_directory(dir_path, model, prefix, skip_CHI, drop_rate_CHI=drop_rate_CHI)
    return result

# Create Gradio Blocks with custom CSS
demo = gr.Blocks(css=css)
with demo:
    # Add header and title
    with gr.Row():
        gr.Markdown("<div class='header'><h1 class='title'>CaptionFlow</h1></div>")
        gr.Markdown("<div class='repo-link'><a href='https://github.com/lrzjason/CaptionFlow'>View on GitHub</a></div>")
    
    # Input parameters section
    with gr.Row():
        with gr.Column():
            gr.Markdown("## You must have an API key to use this service. Either enter here or config in api.json")
            gr.Markdown("## Request API Keys at <a href='https://console.mistral.ai/api-keys/'>Mistral</a>")
            
            api_key = gr.Textbox(label="API Key", placeholder="Leave empty to use api.json", elem_classes="input-box")
       
    with gr.Row():     
        with gr.Column():
            dir_path = gr.Textbox(label="Directory Path", elem_classes="input-box")
        with gr.Column():
            prefix = gr.Textbox(label="Prefix", elem_classes="input-box")
    with gr.Row():  
            selected_chi = gr.CheckboxGroup(choices=chi_options, label="The generation would exclude the following CHI Types if checked.", elem_classes="input-box")
            drop_rate_CHI = gr.Slider(minimum=0, maximum=1, step=0.01, label="Drop Rate for CHI", value=0.1, elem_classes="input-box")
            submit_button = gr.Button("Generate Captions")
    
    # Output section
    with gr.Row():
        output_text = gr.Textbox(label="Status", elem_classes="output-box")
    
    # Connect the button click to the function
    submit_button.click(save_config, inputs=[dir_path, prefix, api_key, selected_chi, drop_rate_CHI], outputs=None)
    submit_button.click(caption_directory, 
                        inputs=[dir_path, prefix, api_key, selected_chi, drop_rate_CHI], 
                        outputs=output_text)

# Launch the app
demo.launch()