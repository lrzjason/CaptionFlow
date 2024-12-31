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
    # input_dir = "C:/Users/lrzja/Desktop/image_caption"
    # text_emb_ext = ".npsd35"
    # max_attempt_count = 3
    model = Phi35ModelWrapper()
    user_prompt = "Lind@V1,full body view,dynamic angles,photoshoot poses; the image is a portrait of a woman, she is sitting at a table, drinking a cup of coffee. On the coffee cup it says \"I love civitai!\".  She has dark hair styled in loose curls. she has piercing brown eyes. she has the perfect milf body with wide hips, and huge breasts. She is wearing a white off-the-shoulder blouse with ruffled sleeves and blue jeans. Her long dark hair is styled in loose waves and falls over her shoulders. She has a slight smile on her face and is looking directly at the camera. The background is blurred, but it appears to be a busy restaurant with people sitting at tables and chairs. The lighting is soft and natural, creating a warm and inviting atmosphere. The overall mood of the image is happy and relaxed.RAW candid cinema,16mm,color graded portra 400 film,remarkable color,ultra realistic,textured skin,remarkable detailed pupils,realistic dull skin noise,visible skin detail,skin fuzz,dry skin,shot with cinematic camera,detailed skin texture,(blush:0.2),(goosebumps:0.3),subsurface scattering,beautiful photograph in the style of Augustus John,Sergio Toppi,Virginia Frances Sterrett,8k HD,detailed skin texture,ultra realistic,textured skin,analog raw photo,cinematic grain,whimsical,"
    result = model.execute(prompt=user_prompt)
    print(result)
    # files = glob.glob(f"{input_dir}/**", recursive=True)
    # image_exts = [".png",".jpg",".jpeg",".webp"]
    # image_files = [f for f in files if os.path.splitext(f)[-1].lower() in image_exts]
    
    # caption_ext = ".txt"
    
    # text_files = []
    # for image_file in tqdm(image_files,position=0):
    #     text_file = os.path.splitext(image_file)[0] + caption_ext
    #     text_emb_file = os.path.splitext(image_file)[0] + text_emb_ext
    #     # backup text_file
    #     if os.path.exists(text_file):
    #         shutil.copy(text_file, os.path.splitext(image_file)[0] + ".bak.txt")


    # for image_file in tqdm(image_files,position=1):
    #     text_file = os.path.splitext(image_file)[0] + caption_ext
    #     attempt_count = 0
        
    #     # read text_file in to var user_prompt
    #     try:
    #         with open(text_file, "r", encoding="utf-8") as f:
    #             user_prompt = f.read()
    #     except:
    #         print(f"Error: {text_file} not found")
    #         # continue
        
    #     # result = model.execute(prompt=user_prompt)
    #     result = model.execute(prompt=user_prompt)
    #     if " sorry" in result:
    #         while " sorry" in result and attempt_count < max_attempt_count:
    #             result = model.execute(prompt=user_prompt)
    #             attempt_count = attempt_count + 1
    #     new_content = result.strip()
    #     # new caption
    #     with open(text_file, "w", encoding="utf-8") as new_f:
    #         new_f.write(new_content)
    #         print("save new caption: ", text_file)
        
    #     if os.path.exists(text_emb_file):
    #         # remove emb file
    #         os.remove(text_emb_file)