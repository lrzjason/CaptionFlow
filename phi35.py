import requests
import torch
from PIL import Image
from io import BytesIO

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from ModelWrapper import ModelWrapper
from utils import flush,get_device

class Phi35ModelWrapper(ModelWrapper):

    def __init__(self,device=None,dtype=None):
        super().__init__()
        self.device = get_device(device)
        self.model_repo_id = "microsoft/Phi-3.5-mini-instruct"
        if dtype == None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        # self.processor = AutoProcessor.from_pretrained(self.model_repo_id)
        
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
        
        
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
    model = Phi35ModelWrapper()
    user_prompt = "low angle, 1girl, solo, cute crystal transparent ghost, sitting on sofa, in the living room, holding a mobiel phone, halloween atmosphere, pumpkin,  black wings, feathers, black dress, white hairs, looking back in shock,"

    # user_prompt = "25 year old ginger woman standing at the gym wearing black spandex sports bra with deep cleavage and tight spandex black shorts, large natural heavy breasts, freckles, braided, expressiveh, ponytail, smiling, full-body view, small abs"
    
    # prompt = prompt+user_prompt
    result = model.execute(prompt=user_prompt)
    print(result)
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