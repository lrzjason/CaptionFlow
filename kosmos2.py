import requests
import torch
from PIL import Image

from transformers import AutoProcessor, Kosmos2ForConditionalGeneration

from ModelWrapper import ModelWrapper
from utils import flush,get_device

class Kosmos2ModelWrapper(ModelWrapper):
    def __init__(self,device=None,dtype=None):
        super().__init__()
        self.device = get_device(device)
        self.model_repo_id = "microsoft/kosmos-2-patch14-224"
        self.processor = AutoProcessor.from_pretrained(self.model_repo_id)
        self.prompt = "<grounding> An image of"
    
    def create(self):
        model = Kosmos2ForConditionalGeneration.from_pretrained(self.model_repo_id).to(self.device)
        return model

    def execute(self,model,image=None,prompt=None):
        if prompt is not None:
            self.prompt = prompt
        processor = self.processor
        inputs = processor(text=self.prompt, images=image, return_tensors="pt").to(self.device)

        generated_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=64,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)
        caption, _ = processor.post_process_generation(generated_text)
        
        
        # clear memory
        del generated_text,inputs,generated_ids
        flush()
        
        return caption
        
    