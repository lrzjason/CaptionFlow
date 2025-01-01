import gradio as gr
import json
from mistralAPI_CHI import get_result, MistralAPIWrapper
import os
import glob
from wd14caption import WD14ModelWrapper
from PIL import Image

# Define default config values
default_config = {
    "dir_path": "",
    "prefix": "",
    "api_key": "",
    "selected_chi": [],
    "drop_rate_CHI": 0.1,
    "use_wd14_fusion":True,
    "wd14_tag_threshold":0.7,
    "prepend_wd14": True,
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
    
    
def save_config(dir_path, prefix, api_key, selected_chi, drop_rate_CHI, use_wd14_fusion, wd14_tag_threshold, prepend_wd14):
    config = {
        "dir_path": dir_path,
        "prefix": prefix,
        "api_key": api_key,
        "selected_chi": selected_chi,
        "drop_rate_CHI": drop_rate_CHI,
        "use_wd14_fusion": use_wd14_fusion,
        "wd14_tag_threshold": wd14_tag_threshold,
        "prepend_wd14": prepend_wd14,
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
def process_directory(dir_path, prefix, api_key, skip_CHI, drop_rate_CHI=0.1, use_wd14_fusion=True, wd14_tag_threshold=0.7, prepend_wd14=True):
    try:
        model = MistralAPIWrapper(api_key=api_key)
    except Exception as e:
        return f"Error initializing model: {e}"
    if not os.path.isdir(dir_path):
        return "Invalid directory path."
    try:
        wd14Model = WD14ModelWrapper()
    except Exception as e:
        print(f"Error initializing WD14 model: {e}")
        use_wd14_fusion = False
        
    image_exts = [".png", ".jpg", ".jpeg", ".webp"]
    image_files = [f for f in glob.glob(os.path.join(dir_path, "**", "*"), recursive=True) if os.path.splitext(f)[-1].lower() in image_exts]
    if not image_files:
        return "No image files found in the directory."
    for image_file in image_files:
        ext = os.path.splitext(image_file)[-1].lower()
        text_file = image_file.replace(ext, ".txt")
        if os.path.exists(text_file):
            print(f"Skipping {image_file} because {text_file} already exists.")
            continue
        
        if use_wd14_fusion:
            image_dir = os.path.dirname(image_file)
            basename = os.path.basename(image_file)
            filename, ext = os.path.splitext(basename)
            json_path = os.path.join(image_dir, f"{filename}.json")
            json_content = {}
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    json_content = json.load(f)
            if "WD14" not in json_content:
                image = Image.open(image_file).convert('RGB')
                result,gender_tags,character_tags = wd14Model.execute(image, tag_threshold=wd14_tag_threshold, character_threshold=wd14_tag_threshold)
                json_content["WD14"] = result
                # save json
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(json_content, f, ensure_ascii=False, indent=4)
            else:
                result = json_content["WD14"]
        caption_text = get_result(image_file, model, prefix, skip_CHI, drop_rate_CHI, tags_information=result)
        if prepend_wd14:
            caption_text = f"{result}; {caption_text}"
        caption_file = os.path.join(image_dir, f"{filename}.txt")
        with open(caption_file, "w", encoding="utf-8") as f:
            f.write(caption_text)
    # save config after execution
    save_config(dir_path, prefix, api_key, skip_CHI, drop_rate_CHI, use_wd14_fusion, wd14_tag_threshold, prepend_wd14)
    return f"Processed {len(image_files)} images in {dir_path}"

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
        selected_chi = gr.CheckboxGroup(choices=chi_options, label="The following CHI Types would be excluded if checked.", elem_classes="input-box")
        drop_rate_CHI = gr.Slider(minimum=0, maximum=1, step=0.01, label="Drop Rate for CHI", value=0.1, elem_classes="input-box")
    
    with gr.Row():
        with gr.Column():
            use_wd14_fusion = gr.Checkbox(label="Use WD14 Fusion", value=True, elem_classes="input-box")
            wd14_tag_threshold = gr.Slider(minimum=0, maximum=0.99, step=0.01, label="WD14 threshold", value=0.7, elem_classes="input-box")
        with gr.Column():
            prepend_wd14 = gr.Checkbox(label="Prepend WD14 result to final caption", value=True, elem_classes="input-box")
    with gr.Row():  
        submit_button = gr.Button("Generate Captions")
    
    # Output section
    with gr.Row():
        output_text = gr.Textbox(label="Status", elem_classes="output-box")
    
    # Connect the button click to the function
    # submit_button.click(save_config, inputs=[dir_path, prefix, api_key, selected_chi, drop_rate_CHI, use_wd14_fusion, wd14_tag_threshold, prepend_wd14], outputs=None)
    submit_button.click(process_directory, 
                        inputs=[dir_path, prefix, api_key, selected_chi, drop_rate_CHI, use_wd14_fusion, wd14_tag_threshold, prepend_wd14], 
                        outputs=output_text)

# Launch the app
demo.launch()