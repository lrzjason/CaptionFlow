
from ModelWrapper import ModelWrapper
from utils import flush,get_device

from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import torch

from PIL import Image

class LlavaSomModelWrapper(ModelWrapper):
    def __init__(self,device=None,dtype=None,tokenizer_repo_id=None):
        super().__init__()
        self.device = get_device(device)
        self.model_repo_id = "zzxslp/som-llava-v1.5-13b-hf"
        if tokenizer_repo_id == None:
            self.tokenizer_repo_id = self.model_repo_id
        if dtype == None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(self.model_repo_id)
        
        # self.prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
        self.prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self.dtype,
            bnb_4bit_quant_type="fp4",
        )
        model = LlavaForConditionalGeneration.from_pretrained(self.model_repo_id, quantization_config=quantization_config, low_cpu_mem_usage=True) 
        # model.to(self.device)
        self.model = model
    
    def execute(self,image=None):
        model = self.model
        processor = self.processor
        # image = Image.open(requests.get(url, stream=True).raw)
        prompt = self.prompt

        inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

        # autoregressively complete prompt
        output = model.generate(**inputs, max_new_tokens=200)
        response = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # response = processor.decode(output[0], skip_special_tokens=True)
        prompt = prompt.replace(" <image>"," ")
        response = response.replace(prompt,'').strip()
        
        return response
    
if __name__ == "__main__":
    image_path = "12.webp"
    image = Image.open(image_path)
    model = LlavaSomModelWrapper()
    result = model.execute(image)
    print(result)