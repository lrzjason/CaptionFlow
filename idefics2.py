import requests
import torch
from PIL import Image
from io import BytesIO

from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

from ModelWrapper import ModelWrapper
from utils import flush,get_device

class Idefics2ModelWrapper(ModelWrapper):

    def __init__(self,device=None,dtype=None):
        super().__init__()
        self.device = get_device(device)
        self.model_repo_id = "HuggingFaceM4/idefics2-8b"
        if dtype == None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(self.model_repo_id)
    
    def create(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self.dtype,
            bnb_4bit_quant_type="fp4",
        )
        model = AutoModelForVision2Seq.from_pretrained(
            self.model_repo_id,
            low_cpu_mem_usage=True,
            torch_dtype=self.dtype,
            # load_in_4bit=True,
            # bnb_4bit_compute_dtype=self.dtype
            quantization_config=quantization_config
        ).eval()
        # .to(self.device)
        return model

    def execute(self,model,image=None):
        # Create inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image in detail."},
                ]
            },   
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}


        # Generate
        # with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
        generated_ids = model.generate(**inputs, max_new_tokens=500)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        spliter = "\nAssistant: "
        generated_texts = generated_texts[generated_texts.index(spliter)+len(spliter):]
        
        
        # clear memory
        del generated_ids,inputs
        flush()
        
        return generated_texts
    