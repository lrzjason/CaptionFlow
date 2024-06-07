import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ModelWrapper import ModelWrapper
from utils import flush,get_device


class Glm4ModelWrapper(ModelWrapper):
    def __init__(self,device=None,dtype=None,tokenizer_repo_id=None):
        super().__init__()
        self.device = get_device(device)
        self.model_repo_id = "THUDM/glm-4-9b-chat"
        if tokenizer_repo_id == None:
            self.tokenizer_repo_id = self.model_repo_id
        if dtype == None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_repo_id, trust_remote_code=True)
        
        self.query = ("This is a hard problem. Carefully summarize in ONE precisely, detailed caption with every element. "
                        "Include composition, angle and perspective based on the following multiple captions "
                        "by different (possibly incorrect) people describing the same scene. "
                        "Be sure to describe everything, and avoid hallucination. Caption:{}")
        
        self.gen_kwargs = {
            "max_length": 2500, 
            "do_sample": True, 
            "top_k": 1,
            # 'repetition_penalty': 1.15
        } 
    
    def create(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_repo_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.device).eval()
        return model

    def execute(self,model,image=None,query=None, captions=""):
        if query != None:
            self.query = query
        tokenizer = self.tokenizer
        query_with_captions = self.query.format(captions)
        # print("query_with_captions\n",query_with_captions)
        inputs = tokenizer.apply_chat_template([{"role": "user", "content": query_with_captions}],
                                            add_generation_prompt=True,
                                            tokenize=True,
                                            return_tensors="pt",
                                            return_dict=True
                                            ).to(self.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, **self.gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # clear memory
        del outputs,inputs
        flush()
        
        return response
    
    