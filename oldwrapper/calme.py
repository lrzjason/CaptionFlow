import requests
import torch
from PIL import Image
from io import BytesIO

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from ModelWrapper import ModelWrapper
from utils import flush,get_device
import glob
from tqdm import tqdm
import shutil
import os

class Phi35ModelWrapper(ModelWrapper):

    def __init__(self,device=None,dtype=None):
        super().__init__()
        self.device = get_device(device)
        self.model_repo_id = "MaziyarPanahi/calme-2.1-phi3.5-4b"
        if dtype == None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        # self.processor = AutoProcessor.from_pretrained(self.model_repo_id)
        
        self.tokenizer = AutoTokenizer.from_pretrained("MaziyarPanahi/calme-2.1-phi3.5-4b")
        
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_repo_id,
            # low_cpu_mem_usage=True,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).eval().to(self.device)
        
        
        chi_txt = "F:/CaptionFlow/CHI.txt"
        # read chi file in to var prompt
        with open(chi_txt, "r", encoding="utf-8") as f:
            prompt = f.read()
        # print("prompt:")
        # print(prompt)
        self.system_prompt = prompt
    

    def execute(self,prompt=""):
        model = self.model
        tokenizer = self.tokenizer
        # Create inputs
        messages = [
            {
                "role": "system", 
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": prompt
            },   
        ]
        # messages = [
        #     {"role": "user", "content": prompt},
        # ]
        # input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to("cuda")

        # # Generate
        # # with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
        
        # outputs = model.generate(**input_ids, max_new_tokens=256)
        # generated_texts = tokenizer.decode(outputs[0])
        
        # return generated_texts
        
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device="cuda"
        )

        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

        output = pipe(messages, **generation_args)
        # print(output[0]['generated_text'])
        
        return output[0]['generated_text']
    

if __name__ == "__main__":
    input_dir = "F:/ImageSet/IC_LORA_DIFF_TRAIN/process_png"
    text_emb_ext = ".npsd35"
    max_attempt_count = 3
    model = Phi35ModelWrapper()
    # user_prompt = "This image is a vibrant and colorful illustration in the style of Japanese anime. It features a youthful character with striking, large circular eyeglasses that have a soft pink tint. The character's hair is styled into two large, symmetrical buns wrapped with blue, white, and pink striped ribbon, giving a whimsical candy aesthetic. Each bun is adorned with a flower-shaped accessory. Their hair is blonde and partially covered by a white headband, which has a pink outline and a motif that continues the candy theme with a small flower and a candy-like charm.  The character's skin glows with a soft, light complexion and is adorned with shining highlights that suggest a glossy texture, possibly from the bubbles surrounding them. They wear a white lab coat over a pink and light blue outfit, with a candy heart emblem on the lapel, which adds to the overall sweet and playful theme. The coat's cuffs are large and fluffy, enhancing the character's youthful charm. Light blue gloves cover their hands.  The character holds a transparent heart-shaped container filled with brightly colored candies and gummies, which add to the image's whimsical and fantastical theme. Multi-colored streaks and frosted bubbles float around, creating a magical atmosphere. Additionally, one of the character's hands is playfully positioned by their mouth, as if blowing a kiss or tasting one of the candies.  The background is awash with pastel colors blending into each other, punctuated by motifs such as stars, hearts, and bubbles that reinforce the sweet and enchanting setting. Two additional elements, a lollipop with a rainbow swirl and a yin-yang symbol in one of the lower corners, introduce interesting visual contrasts to the composition.  The character's expression is one of innocent delight, with a slight blush on their cheeks, and the twinkling in their eye adds to the joyful ambiance of the image. The overall impression is one of a playful, candy-themed fantasy."

    files = glob.glob(f"{input_dir}/**", recursive=True)
    image_exts = [".png",".jpg",".jpeg",".webp"]
    image_files = [f for f in files if os.path.splitext(f)[-1].lower() in image_exts]
    
    caption_ext = ".txt"
    
    text_files = []
    for image_file in tqdm(image_files,position=2):
        text_file = os.path.splitext(image_file)[0] + caption_ext
        text_emb_file = os.path.splitext(image_file)[0] + text_emb_ext
        # backup text_file
        if os.path.exists(text_file):
            shutil.copy(text_file, os.path.splitext(image_file)[0] + ".bak.txt")


    for image_file in tqdm(image_files,position=2):
        text_file = os.path.splitext(image_file)[0] + caption_ext
        attempt_count = 0
        
        # read text_file in to var user_prompt
        try:
            with open(text_file, "r", encoding="utf-8") as f:
                user_prompt = f.read()
        except:
            print(f"Error: {text_file} not found")
            continue
        
        result = model.execute(prompt=user_prompt)
        if " sorry" in result:
            while " sorry" in result and attempt_count < max_attempt_count:
                result = model.execute(image_path=image_file)
                attempt_count = attempt_count + 1
        new_content = result.strip()
        # new caption
        with open(text_file, "w", encoding="utf-8") as new_f:
            new_f.write(new_content)
            print("save new caption: ", text_file)
        
        # if os.path.exists(text_emb_file):
        #     # remove emb file
        #     os.remove(text_emb_file)