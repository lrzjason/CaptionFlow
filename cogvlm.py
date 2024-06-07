"""
This is a demo for using CogAgent and CogVLM in CLI
Make sure you have installed vicuna-7b-v1.5 tokenizer model (https://huggingface.co/lmsys/vicuna-7b-v1.5), full checkpoint of vicuna-7b-v1.5 LLM is not required.
In this demo, We us chat template, you can use others to replace such as 'vqa'.
Strongly suggest to use GPU with bfloat16 support, otherwise, it will be slow.
Mention that only one picture can be processed at one conversation, which means you can not replace or insert another picture during the conversation.
"""

import argparse
import torch

from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer, BitsAndBytesConfig
import os
import time
from ModelWrapper import ModelWrapper
from utils import flush,get_device

class CogvlmModelWrapper(ModelWrapper):

    def __init__(self,device=None,dtype=None,tokenizer_repo_id="lmsys/vicuna-7b-v1.5"):
        super().__init__()
        self.device = get_device(device)
        self.model_repo_id = "THUDM/cogvlm-chat-hf"
        self.tokenizer_repo_id = tokenizer_repo_id
        if dtype == None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.tokenizer = LlamaTokenizer.from_pretrained(self.tokenizer_repo_id,padding_side="right")
        
        self.gen_kwargs = {
            'min_new_tokens':100,
            'max_new_tokens':350,
            'num_beams':1,
            'length_penalty':1,
            'top_k':60,
            'top_p':0.6,
            'repetition_penalty': 1.15,
            'no_repeat_ngram_size':0,
            "do_sample": True,
            "temperature": 0.6,
        } 
        self.prompt = f'Describe the image precisely, detailing every element, interaction and background. Include composition, angle and perspective. Use only facts and concise language; avoid interpretations or speculation:'
        self.starts_with = f'The image showcases '
        
    def get_device(self,device):
        if device == None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        return self.device
    
    def create(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self.dtype,
            bnb_4bit_quant_type="fp4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_repo_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            # load_in_4bit=True,
            # bnb_4bit_compute_dtype=self.dtype
            quantization_config=quantization_config,
            trust_remote_code=True
        ).eval()
        # .to(self.device)
        return model

    def execute(self, model,image=None,prompt=None,starts_with=None):
        if prompt != None:
            self.prompt = prompt
        if starts_with != None:
            self.starts_with = starts_with
        
        torch_type = self.dtype
        query = f'Question: {self.prompt} Answer: {self.starts_with}'
        history = []
        input_by_model = model.build_conversation_input_ids(self.tokenizer, query=query, history=history, images=[image])

        prepare_images = []
        if self.gen_kwargs['num_beams'] > 1:
            prepare_images = [[input_by_model['images'][0].to(self.device).to(torch_type)] for _ in range(self.gen_kwargs['num_beams'])]
        else:
            prepare_images = [[input_by_model['images'][0].to(self.device).to(torch_type)]] if image is not None else None
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(self.device),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(self.device),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(self.device),
            'images': prepare_images,
        }
        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [[input_by_model['cross_images'][0].to(self.device).to(torch_type)]]

        response = ""
        with torch.no_grad():
            outputs = model.generate(**inputs, **self.gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0])
            response = response.split("</s>")[0]
            response = replace(response)
        
        # clear memory
        del outputs,inputs,prepare_images,input_by_model
        flush()
        
        return response
def replace(response):
    # trancate hallucination 
    if "Answer:" in response:
        response = response[:response.index("Answer:")]

    if "watermark" in response:
        start = response.find("watermark")
        sentence_start = response.rfind('.', 0, start) + 1
        response = response[:sentence_start]

    if "caption " in response:
        start = response.find("caption ")
        sentence_start = response.rfind('.', 0, start) + 1
        response = response[:sentence_start]

    if "signature " in response:
        start = response.find("signature ")
        sentence_start = response.rfind('.', 0, start) + 1
        response = response[:sentence_start]

    if "signed by " in response:
        start = response.find("signed by ")
        sentence_start = response.rfind('.', 0, start) + 1
        response = response[:sentence_start]

    if "bottom right corner" in response:
        start = response.find("bottom right corner")
        sentence_start = response.rfind('.', 0, start) + 1
        response = response[:sentence_start]
    return response
