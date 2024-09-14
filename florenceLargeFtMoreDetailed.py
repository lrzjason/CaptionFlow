
from ModelWrapper import ModelWrapper
from utils import flush,get_device

# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoProcessor, AutoModelForCausalLM 
import torch

from PIL import Image
import os

class FlorenceLargeFtMoreDetailedModelWrapper(ModelWrapper):
    def __init__(self,device=None,dtype=None,tokenizer_repo_id=None):
        super().__init__()
        self.device = get_device(device)
        self.model_repo_id = "yayayaaa/florence-2-large-ft-moredetailed"
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
        self.query = "<MORE_DETAILED_CAPTION>"
        model = AutoModelForCausalLM.from_pretrained(self.model_repo_id, trust_remote_code=True)
        model.to(self.device)
        self.model = model
    
    def execute(self,image=None,query=None,captions=""):
        model = self.model
        processor = self.processor
        # image = Image.open(requests.get(url, stream=True).raw)
        
        if query != None:
            self.query = query
        query_with_captions =self.query
        if len(captions)>0:
            query_with_captions = self.query.format(captions)

        inputs = processor(query_with_captions, image, return_tensors="pt").to(self.device)

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

        response = processor.post_process_generation(generated_text, task=self.query, image_size=(image.width, image.height))
        response = response[self.query]
        # print(response)
        
        return response
    
if __name__ == "__main__":
    # image_path = "F:/ImageSet/sd3_test/1_creative_photo/ComfyUI_temp_zpsmu_00236_.png"
    # image = Image.open(image_path)
    model = FlorenceLargeFtMoreDetailedModelWrapper()
    input_dir = "F:/ImageSet/niji"
    # loop input_dir for each image
    for image_path in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_path)
        text_file = os.path.splitext(image_path)[0] + ".txt"
        if not image_path.endswith(".png"): continue
        if os.path.exists(text_file): 
            print("skip exists: ", text_file)
            continue
        image_path = os.path.join(input_dir, image_path)
        print(image_path)
        image = Image.open(image_path)
        result = model.execute(image)
        
        # save result as txt with the same name as image file without extension
        result_path = os.path.join(input_dir, text_file)
        with open(result_path, "w") as f: f.write(result)
        print(result_path)
        print(result)
        flush()