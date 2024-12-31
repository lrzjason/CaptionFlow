import base64
import requests
import os
from mistralai import Mistral
from ModelWrapper import ModelWrapper

# read api from api.json
import json
from tqdm import tqdm
import glob
import time
from io import BytesIO
from PIL import Image
import math
import random
def encode_image(image_path):
    """Encode the image to base64."""
    try:
        # with open(image_path, "rb") as image_file:
        #     return base64.b64encode(image_file.read()).decode('utf-8')
        
        # open image and scale to 1 mega pixels
        image = Image.open(image_path)
        # scale to 1 mega pixels with same aspect ratio
        # Get the original dimensions
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        # Calculate the scaling factor to make the image ~1,000,000 pixels
        target_pixels = 1024 * 1024  # 1 Megapixel
        scale_factor = math.sqrt(target_pixels / (original_width * original_height))

        # Compute new dimensions while maintaining the aspect ratio
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        # Resize the image using the calculated dimensions
        image = image.resize((new_width, new_height), Image.LANCZOS)
        # show image
        # image.show()
        
        # Encode image to base64
        buffered = BytesIO()
        image = image.convert("RGB")
        image.save(buffered, format="JPEG")  # Save as JPEG in memory
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Return the base64 encoded string
        return img_base64
        
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:  # Added general exception handling
        print(f"Error: {e}")
        return None


class MistralAPIWrapper(ModelWrapper):

    def __init__(self,device=None,dtype=None,tokenizer_repo_id="lmsys/vicuna-7b-v1.5"):
        super().__init__()
        
        with open("api.json", "r") as f:
            api_data = json.load(f)
            
        api_key = api_data["MISTRAL_API_KEY"]
        self.client = Mistral(api_key=api_key)
        
        self.model = "pixtral-12b-2409"
        
        # self.prompt = "Please describe the image in details and but only in one paragraph."
        
        # self.sys_prompt = "You are a professional color artist and painting expert. You have a deep understanding of how colors are used in visual art to create harmony, contrast, mood, and balance. You can analyze the color composition of an image and describe it in detailed, artistic language, as a trained artist would."
# System Message (if your interface supports it):
# "You are a professional color artist and painting expert. You have a deep understanding of how colors are used in visual art to create harmony, contrast, mood, and balance. You can analyze the color composition of an image and describe it in detailed, artistic language, as a trained artist would."

        
        chi_json = "F:/CaptionFlow/CHI.json"
        # read chi file in to var prompt
        # with open(chi_txt, "r", encoding="utf-8") as f:
        #     self.prompt = f.read()
        
        chi_type = "CHI_BLUR"
        
        # load json file
        with open(chi_json, "r", encoding="utf-8") as f:
            self.chi_json = json.load(f)
            self.sys_prompt = self.chi_json[chi_type]["system_prompt"]
            self.prompt = self.chi_json[chi_type]["user_prompt"]
    def execute(self, image=None,sys_prompt=None,prompt=None,starts_with=None, image_path=None):
        if prompt != None:
            self.prompt = prompt
        if sys_prompt != None:
            self.sys_prompt = sys_prompt
        base64_image = encode_image(image_path)
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": self.sys_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}" 
                    }
                ]
            }
        ]
        chat_response = self.client.chat.complete(
            model=self.model,
            messages=messages
        )
        return chat_response.choices[0].message.content

       
if __name__ == "__main__":
    # F:/ImageSet/kolors_cosplay/train_chibi/chibi
    # input_dir = "F:/ImageSet/kolors_cosplay/train_vector/vector_person"
    
    # input_dir = "F:/ImageSet/kolors_cosplay/train/azami_face"
    # input_dir = "F:/ImageSet/flux/ic-lora-restore/raw/test_old"
    # F:\ImageSet\IC_LORA_DIFF_TRAIN\SIDE_BY_SIDE\new
    model = MistralAPIWrapper()
    # prefix = "二次元动漫风格, anime artwork, chibi, "
    prefix = ""
    max_attempt_count = 3
    character = ""
    # image_file = "E:/Media/0AIpainting/20241205/72d92a9269304d0af753a7085559937f.jpg"
    # image_file = "E:/Media/0AIpainting/20241209/1.jpg"
    # image_file = "E:/Media/0AIpainting/20241205/5.webp"
    # prompt = "Evaluate the domaint colors of the image. List 5 colors from important to less important. Return a list using JSON format. Each items should have color name and confidence."
    # result = model.execute(image_path=image_file)
    # print(result)
    # encode_image(image_file)
    
    # input_dir = "F:/ImageSet/flux/cutecollage"
    input_dir = "F:/ImageSet/flux/temp"
    
    files = glob.glob(f"{input_dir}/**", recursive=True)
    image_exts = [".png",".jpg",".jpeg",".webp"]
    image_files = [f for f in files if os.path.splitext(f)[-1].lower() in image_exts]
    
    # caption_ext = ".vl2"
    caption_ext = ".txt"
    prefix = "cutecollage, "
    
    def caption(image_file,sys_prompt,prompt):
        attempt_count = 0
        try:
            result = model.execute(image_path=image_file,sys_prompt=sys_prompt,prompt=prompt)
            if " sorry" in result:
                while " sorry" in result and attempt_count < max_attempt_count:
                    result = model.execute(image_path=image_file,sys_prompt=sys_prompt,prompt=prompt)
                    attempt_count = attempt_count + 1
        except:
            # sleep 5 seconds
            time.sleep(5)
            result = model.execute(image_path=image_file)
            if " sorry" in result:
                while " sorry" in result and attempt_count < max_attempt_count:
                    result = model.execute(image_path=image_file)
                    attempt_count = attempt_count + 1
        return result
    
    def get_result(image_file, prefix="",skip_CHI=["CHI_THREE_SECTION"]):
        basename = os.path.basename(image_file)
        image_dir = os.path.dirname(image_file)
        filename, ext = os.path.splitext(basename)
        result = {}
        result_path = f"{image_dir}/{filename}.json"
        # read result_path
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as json_file:
                result = json.load(json_file)

        new_result = {}
        drop_rate = 0.1
        for key in model.chi_json.keys():
            if key != "CHI_SUBJECT" and random.random() < drop_rate:
                continue
            if key in result:
                new_result[key] = result[key]
                continue
            if not key in skip_CHI:
                print(key)
                new_result[key] = caption(image_file,model.chi_json[key]["system_prompt"],model.chi_json[key]["user_prompt"])
            
        result = new_result
        # serialize result to string
        result_str = json.dumps(result)
        summary_result = model.chi_json["CHI_SUMMARY"]["user_prompt"] + result_str
        result["CHI_SUMMARY"] = caption(image_file,model.chi_json["CHI_SUMMARY"]["system_prompt"],summary_result)
        
        # dump result to json
        with open(result_path, "w", encoding="utf-8") as json_file:
            json.dump(result, json_file, ensure_ascii=False, indent=4)
        
        # save to caption file
        caption_file = f"{image_dir}/{filename}{caption_ext}"
        text = prefix + result["CHI_SUMMARY"]
        with open(caption_file, "w", encoding="utf-8") as f:
            f.write(text)
     
    skip_CHI = ["CHI_THREE_SECTION"]
    for image_file in tqdm(image_files):
        # complete following code
        result = get_result(image_file, prefix, skip_CHI)
        
        