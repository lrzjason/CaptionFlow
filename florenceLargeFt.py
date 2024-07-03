
from ModelWrapper import ModelWrapper
from utils import flush,get_device

# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoProcessor, AutoModelForCausalLM 
import torch

from PIL import Image

class FlorenceLargeFtModelWrapper(ModelWrapper):
    def __init__(self,device=None,dtype=None,tokenizer_repo_id=None):
        super().__init__()
        self.device = get_device(device)
        self.model_repo_id = "microsoft/Florence-2-large-ft"
        if tokenizer_repo_id == None:
            self.tokenizer_repo_id = self.model_repo_id
        if dtype == None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(self.model_repo_id, trust_remote_code=True)
        
        # self.prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
        # self.prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nWhat is shown in this image? ASSISTANT:"
        # self.prompt = "<image>Please describe the content of this image."
        self.prompt = "<MORE_DETAILED_CAPTION>"
        model = AutoModelForCausalLM.from_pretrained(self.model_repo_id, trust_remote_code=True)
        model.to(self.device)
        self.model = model
    
    def execute(self,image=None):
        model = self.model
        processor = self.processor
        # image = Image.open(requests.get(url, stream=True).raw)
        prompt = self.prompt

        inputs = processor(prompt, image, return_tensors="pt").to(self.device)

        # # autoregressively complete prompt
        # output = model.generate(**inputs, max_new_tokens=300)
        # response = processor.decode(output[0][2:], skip_special_tokens=True)
        # prompt = prompt.replace("<image>"," ")
        # response = response.replace(prompt,'').strip()
        
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        response = processor.post_process_generation(generated_text, task=self.prompt, image_size=(image.width, image.height))
        response = response[self.prompt]
        # print(response)
        
        return response
    
if __name__ == "__main__":
    image_path = "15.png"
    image = Image.open(image_path)
    llaveNext = FlorenceLargeFtModelWrapper()
    result = llaveNext.execute(image)
    print(result)