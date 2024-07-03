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
        
        self.query = ("{}")
        
        self.gen_kwargs = {
            "max_length": 8000, 
            "do_sample": True, 
            "top_k": 1,
            # 'repetition_penalty': 1.15
        } 
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self.dtype,
            bnb_4bit_quant_type="fp4",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_repo_id,
            # torch_dtype=self.dtype,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.device).eval()
    
    def execute(self,query=None, captions=""):
        model = self.model
        tokenizer = self.tokenizer
        if query != None:
            self.query = query
        query_with_captions =self. query
        if len(captions)>0:
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
    
    