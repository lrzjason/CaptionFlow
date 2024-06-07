import argparse
import torch

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time
from ModelWrapper import ModelWrapper
from utils import flush,get_device

class Cogvlm2ModelWrapper(ModelWrapper):

    def __init__(self,device=None,dtype=None):
        super().__init__()
        self.device = get_device(device)
        self.model_repo_id = "THUDM/cogvlm2-llama3-chat-19B-int4"
        if dtype == None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        # self.tokenizer = LlamaTokenizer.from_pretrained(self.tokenizer_repo_id,padding_side="right")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_repo_id,
            trust_remote_code=True
        )
        # self.gen_kwargs = {
        #     'min_new_tokens':100,
        #     'max_new_tokens':350,
        #     'num_beams':1,
        #     'length_penalty':1,
        #     'top_k':60,
        #     'top_p':0.6,
        #     'repetition_penalty': 1.15,
        #     'no_repeat_ngram_size':0,
        #     "do_sample": True,
        #     "temperature": 0.6,
        # } 
        self.prompt = f'Describe the image precisely, detailing every element, interaction and background. Include composition, angle and perspective. Use only facts and concise language; avoid interpretations or speculation:'
        # self.starts_with = f'The image showcases '
        self.query = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"
        
    def get_device(self,device):
        if device == None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        return self.device
    
    def create(self):
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=self.dtype,
        #     bnb_4bit_quant_type="fp4",
        # )
        # model = AutoModelForCausalLM.from_pretrained(
        #     self.model_repo_id,
        #     torch_dtype=self.dtype,
        #     low_cpu_mem_usage=True,
        #     # load_in_4bit=True,
        #     # bnb_4bit_compute_dtype=self.dtype
        #     quantization_config=quantization_config,
        #     trust_remote_code=True
        # ).eval()
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_repo_id,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            # quantization_config=quantization_config
        ).eval()
        # .to(self.device)
        return model

    def execute(self, model,image=None,prompt=None,starts_with=None):
        if prompt != None:
            self.prompt = prompt
        if starts_with != None:
            self.starts_with = starts_with
        tokenizer = self.tokenizer
        torch_type = self.dtype
        device = self.device
        history = []
        query = self.query.format(self.prompt)
        input_by_model = model.build_conversation_input_ids(
                tokenizer,
                query=query,
                history=history,
                images=[image],
                template_version='chat'
            )
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(device),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(device),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(device),
            'images': [[input_by_model['images'][0].to(device).to(torch_type)]] if image is not None else None,
        }
        gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,
        }
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0])
            response = response.split("<|end_of_text|>")[0]
            print("\nCogVLM2:", response)
        # query = f'Question: {self.prompt} Answer: {self.starts_with}'
        # history = []
        # input_by_model = model.build_conversation_input_ids(self.tokenizer, query=query, history=history, images=[image])

        # prepare_images = []
        # if self.gen_kwargs['num_beams'] > 1:
        #     prepare_images = [[input_by_model['images'][0].to(self.device).to(torch_type)] for _ in range(self.gen_kwargs['num_beams'])]
        # else:
        #     prepare_images = [[input_by_model['images'][0].to(self.device).to(torch_type)]] if image is not None else None
        # inputs = {
        #     'input_ids': input_by_model['input_ids'].unsqueeze(0).to(self.device),
        #     'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(self.device),
        #     'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(self.device),
        #     'images': prepare_images,
        # }
        # if 'cross_images' in input_by_model and input_by_model['cross_images']:
        #     inputs['cross_images'] = [[input_by_model['cross_images'][0].to(self.device).to(torch_type)]]

        # response = ""
        # with torch.no_grad():
        #     outputs = model.generate(**inputs, **self.gen_kwargs)
        #     outputs = outputs[:, inputs['input_ids'].shape[1]:]
        #     response = self.tokenizer.decode(outputs[0])
        #     response = response.split("</s>")[0]
        #     response = replace(response)
        
        # clear memory
        del outputs,inputs,input_by_model
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


# import torch
# from PIL import Image
# from transformers import AutoModelForCausalLM, AutoTokenizer

# MODEL_PATH = "THUDM/cogvlm2-llama3-chat-19B-int4"
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
#     0] >= 8 else torch.float16

# tokenizer = AutoTokenizer.from_pretrained(
#     MODEL_PATH,
#     trust_remote_code=True
# )
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH,
#     torch_dtype=TORCH_TYPE,
#     trust_remote_code=True,
#     low_cpu_mem_usage=True,
# ).eval()

# text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"

# while True:
#     image_path = input("image path >>>>> ")
#     if image_path == '':
#         print('You did not enter image path, the following will be a plain text conversation.')
#         image = None
#         text_only_first_query = True
#     else:
#         image = Image.open(image_path).convert('RGB')

#     history = []

#     while True:
#         query = input("Human:")
#         if query == "clear":
#             break

#         if image is None:
#             if text_only_first_query:
#                 query = text_only_template.format(query)
#                 text_only_first_query = False
#             else:
#                 old_prompt = ''
#                 for _, (old_query, response) in enumerate(history):
#                     old_prompt += old_query + " " + response + "\n"
#                 query = old_prompt + "USER: {} ASSISTANT:".format(query)
#         if image is None:
#             input_by_model = model.build_conversation_input_ids(
#                 tokenizer,
#                 query=query,
#                 history=history,
#                 template_version='chat'
#             )
#         else:
#             input_by_model = model.build_conversation_input_ids(
#                 tokenizer,
#                 query=query,
#                 history=history,
#                 images=[image],
#                 template_version='chat'
#             )
#         inputs = {
#             'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
#             'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
#             'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
#             'images': [[input_by_model['images'][0].to(DEVICE).to(TORCH_TYPE)]] if image is not None else None,
#         }
#         gen_kwargs = {
#             "max_new_tokens": 2048,
#             "pad_token_id": 128002,
#         }
#         with torch.no_grad():
#             outputs = model.generate(**inputs, **gen_kwargs)
#             outputs = outputs[:, inputs['input_ids'].shape[1]:]
#             response = tokenizer.decode(outputs[0])
#             response = response.split("<|end_of_text|>")[0]
#             print("\nCogVLM2:", response)
#         history.append((query, response))

if __name__ == "__main__":
    image_path = "2.webp"
    image = Image.open(image_path)
    cogvlm2 = Cogvlm2ModelWrapper()
    cogvlm2_model = cogvlm2.create()
    result = cogvlm2.execute(cogvlm2_model,image)
    print(result)