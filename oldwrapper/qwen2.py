import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ModelWrapper import ModelWrapper
from utils import flush,get_device


class Qwen2ModelWrapper(ModelWrapper):
    def __init__(self,device=None,dtype=None,tokenizer_repo_id=None):
        super().__init__()
        self.device = get_device(device)
        self.model_repo_id = "Qwen/Qwen1.5-32B-Chat-AWQ"
        if tokenizer_repo_id == None:
            self.tokenizer_repo_id = self.model_repo_id
        # if dtype == None:
        #     self.dtype = torch.float16
        # else:
        #     self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_repo_id)
        
        self.query = ("Give me a short introduction to large language model.")
        self.gen_kwargs = {
            "max_length": 8000, 
            "do_sample": True, 
            "top_k": 1,
            # 'repetition_penalty': 1.15
        } 
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_repo_id,
            torch_dtype="auto",
            device_map="auto"
        )
    
    def execute(self,query=None, captions=""):
        device = self.device
        model = self.model
        if query != None:
            self.query = query
        tokenizer = self.tokenizer
        query_with_captions =self. query
        if len(captions)>0:
            query_with_captions = self.query.format(captions)
        # print("query_with_captions\n",query_with_captions)
        
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": self.query}
        ]
        text = tokenizer.apply_chat_template([
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": self.query}
                    ],
                    tokenize=False,
                    add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=2048
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # with torch.no_grad():
        #     outputs = model.generate(**inputs, **self.gen_kwargs)
        #     outputs = outputs[:, inputs['input_ids'].shape[1]:]
        # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # clear memory
        del outputs,inputs
        flush()
        
        return response
    
    
if __name__ == "__main__":
    # image_path = "12.webp"
    # image = Image.open(image_path)
    model = Qwen2ModelWrapper()
    result = model.execute()
    print(result)