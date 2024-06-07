
from ModelWrapper import ModelWrapper
from utils import flush,get_device

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch

from PIL import Image

class LlavaNextModelWrapper(ModelWrapper):
    def __init__(self,device=None,dtype=None,tokenizer_repo_id=None):
        super().__init__()
        self.device = get_device(device)
        self.model_repo_id = "llava-hf/llava-v1.6-mistral-7b-hf"
        if tokenizer_repo_id == None:
            self.tokenizer_repo_id = self.model_repo_id
        if dtype == None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.processor = LlavaNextProcessor.from_pretrained(self.model_repo_id)
        
        self.prompt = "[INST] <image>\nWhat is shown in this image? [/INST] The image shows "
    
    def create(self):
        model = LlavaNextForConditionalGeneration.from_pretrained(self.model_repo_id, torch_dtype=self.dtype, low_cpu_mem_usage=True) 
        model.to(self.device)
        return model

    def execute(self,model,image=None):
        processor = self.processor
        # image = Image.open(requests.get(url, stream=True).raw)
        prompt = self.prompt

        inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

        # autoregressively complete prompt
        output = model.generate(**inputs, max_new_tokens=200)
        response = processor.decode(output[0], skip_special_tokens=True)
        prompt = prompt.replace("<image>"," ")
        response = response.replace(prompt,'').strip()
        return response
    
if __name__ == "__main__":
    image_path = "2.webp"
    image = Image.open(image_path)
    llaveNext = LlavaNextModelWrapper()
    llaveNext_model = llaveNext.create()
    result = llaveNext.execute(llaveNext_model,image)
    print(result)