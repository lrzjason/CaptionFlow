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

def encode_image(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
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
        
        self.prompt = "Please describe the image in details and but only in one paragraph."
        
        
        chi_json = "F:/CaptionFlow/CHI.json"
        # read chi file in to var prompt
        # with open(chi_txt, "r", encoding="utf-8") as f:
        #     self.prompt = f.read()
        
        chi_type = "CHI_THREE_SECTION"
        
        # load json file
        with open(chi_json, "r", encoding="utf-8") as f:
            self.chi_json = json.load(f)
            self.sys_prompt = self.chi_json[chi_type]["system_prompt"]
            self.prompt = self.chi_json[chi_type]["user_prompt"]

        
    def execute(self, image=None,prompt=None,starts_with=None, image_path=None):
        if prompt == None:
            prompt = self.prompt
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
                        "text": prompt
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
    input_dir = "F:/ImageSet/flux/cutecollage_caption"
    # F:\ImageSet\IC_LORA_DIFF_TRAIN\SIDE_BY_SIDE\new
    model = MistralAPIWrapper()
    # prefix = "二次元动漫风格, anime artwork, chibi, "
    prefix = ""
    max_attempt_count = 3
    character = ""
    
    files = glob.glob(f"{input_dir}/**", recursive=True)
    image_exts = [".png",".jpg",".jpeg",".webp"]
    image_files = [f for f in files if os.path.splitext(f)[-1].lower() in image_exts]
    
    # caption_ext = ".vl2"
    caption_ext = ".pix"
    
    for image_file in tqdm(image_files):
        try:
            text_file = os.path.splitext(image_file)[0] + caption_ext
            if os.path.exists(text_file):
                continue
            attempt_count = 0
            result = model.execute(image_path=image_file)
            if " sorry" in result:
                while " sorry" in result and attempt_count < max_attempt_count:
                    # sleep 5 seconds
                    time.sleep(5)
                    result = model.execute(image_path=image_file)
                    attempt_count = attempt_count + 1
            
            new_content = f"{prefix}{character}{result}"
            # new caption
            with open(text_file, "w", encoding="utf-8") as new_f:
                new_f.write(new_content)
                print("save new caption: ", text_file)
            # break
        except:
            print("exception, might due to network error or request limit")
            # sleep 5 seconds
            time.sleep(5)  
                  
            text_file = os.path.splitext(image_file)[0] + caption_ext
            if os.path.exists(text_file):
                continue
            attempt_count = 0
            result = model.execute(image_path=image_file)
            if " sorry" in result:
                while " sorry" in result and attempt_count < max_attempt_count:
                    # sleep 5 seconds
                    time.sleep(5)
                    result = model.execute(image_path=image_file)
                    attempt_count = attempt_count + 1
            
            new_content = f"{prefix}{character}{result}"
            # new caption
            with open(text_file, "w", encoding="utf-8") as new_f:
                new_f.write(new_content)
                print("save new caption: ", text_file)                           