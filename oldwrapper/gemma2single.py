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
            do_sample=True
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
    user_prompt = "1girl supermodel leans her back against car nvision74 side view inside a sleek futuristic warehouse, some bright LED spotlights shine on it, it is shiny new untouched, medium distance medium shot, two shot angle."
    result = model.execute(prompt=user_prompt)
    print(result)
    