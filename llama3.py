import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ModelWrapper import ModelWrapper
from utils import flush,get_device


class Llama3ModelWrapper(ModelWrapper):
    def __init__(self,device=None,dtype=None,tokenizer_repo_id=None):
        super().__init__()
        self.device = get_device(device)
        self.model_repo_id = "refuelai/Llama-3-Refueled"
        if tokenizer_repo_id == None:
            self.tokenizer_repo_id = self.model_repo_id
        if dtype == None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_repo_id,padding_side="right")
        
        self.query = ("This is a hard problem. Carefully summarize in ONE precisely, detailed caption with every element. "
                        "Include composition, angle and perspective based on the following multiple captions "
                        "by different (possibly incorrect) people describing the same scene. "
                        "Be sure to describe everything, and avoid hallucination. Caption:{}")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_repo_id, torch_dtype=self.dtype, device_map="auto")
        
        

    def execute(self,image=None,query=None,captions=""):
        model = self.model
        tokenizer = self.tokenizer
        if query != None:
            self.query = query
        query_with_captions = self.query.format(captions)
        messages = [{"role": "user", "content": query_with_captions}]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")

        # outputs = model.generate(inputs, **self.gen_kwargs)
        outputs = model.generate(inputs, max_length=2500)
        response = tokenizer.decode(outputs[0])
        response = response.split("<|start_header_id|>assistant<|end_header_id|>")[1].replace("\n","").split("<|eot_id|>")[0]
        
        return response
    