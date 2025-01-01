import base64
from PIL import Image
import os
import glob
import json
import random
import math
from mistralai import Mistral
from ModelWrapper import ModelWrapper
from io import BytesIO

ERROR_MSG = "Caption generation failed."

def encode_image(image_path):
    try:
        image = Image.open(image_path)
        width, height = image.size
        aspect_ratio = width / height
        target_pixels = 1024 * 1024
        scale_factor = math.sqrt(target_pixels / (width * height))
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        image = image.resize((new_width, new_height), Image.LANCZOS)
        buffered = BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

class MistralAPIWrapper(ModelWrapper):
    def __init__(self, api_key=None):
        super().__init__()
        if api_key is None or api_key == "":
            try:
                with open("api.json", "r") as f:
                    api_data = json.load(f)
                    api_key = api_data.get("MISTRAL_API_KEY")
                if api_key is None:
                    raise KeyError("API key not found in api.json.")
            except FileNotFoundError:
                raise FileNotFoundError("api.json file not found.")
            except KeyError as e:
                raise KeyError(str(e))
        self.client = Mistral(api_key=api_key)
        self.model = "pixtral-12b-2409"
        with open("CHI.json", "r", encoding="utf-8") as f:
            self.chi_json = json.load(f)
            self.sys_prompt = self.chi_json["CHI_BLUR"]["system_prompt"]
            self.prompt = self.chi_json["CHI_BLUR"]["user_prompt"]
    
    def execute(self, image_path, sys_prompt, prompt):
        base64_image = encode_image(image_path)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}]}
        ]
        print("Sending request to Mistral API...")
        response = self.client.chat.complete(model=self.model, messages=messages)
        return response.choices[0].message.content

def generate_caption(image_file, model, sys_prompt, prompt):
    attempt = 0
    while attempt < 3:
        try:
            caption = model.execute(image_file, sys_prompt, prompt)
            if "sorry" not in caption:
                return caption
            attempt += 1
        except Exception as e:
            print(f"Error generating caption: {e}")
            attempt += 1
    return ERROR_MSG

def get_result(image_file, model, prefix, skip_CHI, drop_rate_CHI=0.1):
    basename = os.path.basename(image_file)
    image_dir = os.path.dirname(image_file)
    filename, ext = os.path.splitext(basename)
    result_path = os.path.join(image_dir, f"{filename}.json")
    result = {}
    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)
    
    new_result = {}
    for key in model.chi_json.keys():
        # CHI_SUBJECT not skip, summary must skip
        if key != "CHI_SUBJECT" and random.random() < drop_rate_CHI or key == "CHI_SUMMARY":
            continue
        if key in result:
            if not ERROR_MSG in result[key]:
                new_result[key] = result[key]
                continue
        if key not in skip_CHI:
            print(f"Generating {key}...")
            sys_prompt = model.chi_json[key]["system_prompt"]
            user_prompt = model.chi_json[key]["user_prompt"]
            caption_text = generate_caption(image_file, model, sys_prompt, user_prompt)
            new_result[key] = caption_text
    
    result.update(new_result)
    result_str = json.dumps(result)
    summary_prompt = model.chi_json["CHI_SUMMARY"]["user_prompt"] + result_str
    print(f"Generating CHI_SUMMARY...")
    summary = generate_caption(image_file, model, model.chi_json["CHI_SUMMARY"]["system_prompt"], summary_prompt)
    summary = summary.strip().replace("\n", " ")
    result["CHI_SUMMARY"] = summary
    
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    caption_file = os.path.join(image_dir, f"{filename}.txt")
    with open(caption_file, "w", encoding="utf-8") as f:
        f.write(prefix + summary)
    return prefix + summary

def process_single_image(image_file, model, prefix, skip_CHI, drop_rate_CHI=0.1):
    try:
        caption_text = get_result(image_file, model, prefix, skip_CHI, drop_rate_CHI)
        print(f"Processed {image_file}")
        return caption_text
    except Exception as e:
        print(f"Error processing {image_file}: {e}")
        return None

def process_directory(dir_path, model, prefix, skip_CHI, drop_rate_CHI=0.1):
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
        process_single_image(image_file, model, prefix, skip_CHI, drop_rate_CHI)
    return f"Processed {len(image_files)} images in {dir_path}"