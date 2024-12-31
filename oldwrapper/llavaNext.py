
from ModelWrapper import ModelWrapper
from utils import flush,get_device

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch

from PIL import Image

class LlavaNextModelWrapper(ModelWrapper):
    def __init__(self,device=None,dtype=None,tokenizer_repo_id=None):
        super().__init__()
        self.device = get_device(device)
        self.model_repo_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
        if tokenizer_repo_id == None:
            self.tokenizer_repo_id = self.model_repo_id
        if dtype == None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.processor = LlavaNextProcessor.from_pretrained(self.model_repo_id)
        
        # self.prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
        self.prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nWhat is shown in this image? ASSISTANT:"
        model = LlavaNextForConditionalGeneration.from_pretrained(self.model_repo_id, torch_dtype=self.dtype, low_cpu_mem_usage=True) 
        model.to(self.device)
        self.model = model
    
    def execute(self,image=None):
        model = self.model
        processor = self.processor
        # image = Image.open(requests.get(url, stream=True).raw)
        prompt = self.prompt

        inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

        # autoregressively complete prompt
        output = model.generate(**inputs, max_new_tokens=200)
        response = processor.decode(output[0], skip_special_tokens=True)
        prompt = prompt.replace(" <image>"," ")
        response = response.replace(prompt,'').strip()
        return response
    
if __name__ == "__main__":
    image_path = "12.webp"
    image = Image.open(image_path)
    llaveNext = LlavaNextModelWrapper()
    result = llaveNext.execute(image)
    print(result)