import requests
import torch
from PIL import Image
from io import BytesIO

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from ModelWrapper import ModelWrapper
from utils import flush,get_device
from tqdm import tqdm
import glob
import shutil
import os

class Gemma2ModelWrapper(ModelWrapper):

    def __init__(self,device=None,dtype=None):
        super().__init__()
        self.device = get_device(device)
        self.model_repo_id = "google/gemma-2-2b-it"
        if dtype == None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        # self.processor = AutoProcessor.from_pretrained(self.model_repo_id)
        
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_repo_id,
            low_cpu_mem_usage=True,
            torch_dtype=self.dtype,
            do_sample=True,
            device=self.device
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
        messages = [
            # {
            #     "role": "system", 
            #     "content": self.system_prompt
            # },
            {
                "role": "user",
                "content": self.system_prompt+prompt
            },   
        ]
        # Create inputs
        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image"},
        #             {"type": "text", "text": "Describe this image in detail."},
        #         ]
        #     },   
        # ]
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
    model = Gemma2ModelWrapper()
    chi_txt = "F:/CaptionFlow/CHI.txt"
    input_dir = "F:/ImageSet/sd35/anime-with-gpt4v-filter"
    # read chi file in to var prompt
    # with open(chi_txt, "r", encoding="utf-8") as f:
    #     prompt = f.read()
    # print("prompt:")
    # print(prompt)
    # user_prompt = "anime artwork, a cute little witch with cat ears, wearing witch hat, black dress, sitting on a flying broom along side with a cute little black cat, holding a pumkin lantern, in the dark sky with a bright big round moon.一个可爱的小女巫长着猫耳朵，戴着女巫帽，穿着黑色的衣服，坐在飞扫帚上，扫帚上吊着南瓜灯，旁边有一只可爱的小黑猫，在黑暗的天空中有一个明亮的大圆月亮。"

    # user_prompt = "25 year old ginger woman standing at the gym wearing black spandex sports bra with deep cleavage and tight spandex black shorts, large natural heavy breasts, freckles, braided, expressiveh, ponytail, smiling, full-body view, small abs"
    
    # prompt = prompt+user_prompt
    text_emb_ext = ".npsd35"

    files = glob.glob(f"{input_dir}/**", recursive=True)
    image_exts = [".png",".jpg",".jpeg",".webp"]
    image_files = [f for f in files if os.path.splitext(f)[-1].lower() in image_exts]
    
    caption_ext = ".txt"
    
    text_files = []
    for image_file in tqdm(image_files):
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
        
        if os.path.exists(text_emb_file):
            # remove emb file
            os.remove(text_emb_file)
    # prompt = f"List the subject of the following prompt: #content#. return in JSON format avoid explainations. the returned JSON should contain subject as key with an array store top 10 subjects."
    # generated = "A chubby, kawaii, cartoonish, c4d render of a small, round, white cotton-stuffed hamster wearing a black belt with a red ribbon tied around its head. It holds an AK47 rifle in its paws, with a soft, pastel color palette. The hamster is facing the camera with a dazed expression, set against a black background with a white bottom."
    # # prompt = user_prompt
    
    # list_a = prompt.replace('#content#',user_prompt)
    # list_b = prompt.replace('#content#',generated)
    
    # a_result = model.execute(prompt=list_a)
    # print(a_result)
    
    
    # b_result = model.execute(prompt=list_b)
    # print(b_result)
    
    
    # subjects_a = [
    # "毛绒公仔",
    # "小仓鼠",
    # "AK47",
    # "白色空手道黑带服装",
    # "红色绑带",
    # "adorable kawaii",
    # "cartoonish",
    # "卡通造型",
    # "圆润可爱",
    # "萌"
    # ]
    # subjects_b = [
    # "Hamster",
    # "Cartoon",
    # "Cotton-stuffed",
    # "Kawaii",
    # "C4D Render",
    # "AK47",
    # "Rifle",
    # "Black Belt",
    # "Red Ribbon",
    # "Daze"
    # ]