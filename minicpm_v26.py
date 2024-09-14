
from ModelWrapper import ModelWrapper
from utils import flush,get_device
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import torch

from PIL import Image
from utils import flush
import os
import glob
import cv2

from tqdm import tqdm

class MinicpmModelWrapper(ModelWrapper):
    def __init__(self,device=None,dtype=None,tokenizer_repo_id=None):
        super().__init__()
        self.device = get_device(device)
        self.model_repo_id = "openbmb/MiniCPM-V-2_6"
        if tokenizer_repo_id == None:
            self.tokenizer_repo_id = self.model_repo_id
        if dtype == None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_repo_id,trust_remote_code=True)
        
        # self.prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
        self.query = "请用中文描述该图片"
        model = AutoModel.from_pretrained(self.model_repo_id, trust_remote_code=True,  attn_implementation='sdpa', torch_dtype=torch.bfloat16) 
        model = model.eval().cuda()
        self.model = model
    
    def execute(self,image=None,query=None, captions=""):
        model = self.model
        # tokenizer = self.tokenizer
        # image = Image.open(requests.get(url, stream=True).raw)
        # prompt = self.prompt
        
        if query != None:
            self.query = query
        query_with_captions =self.query
        if len(captions)>0:
            query_with_captions = self.query.format(captions)
        messages = [ 
            {"role": "user", "content": [image,query_with_captions]}, 
            # {"role": "assistant", "content": "The chart displays the percentage of respondents who agree with various statements about their preparedness for meetings. It shows five categories: 'Having clear and pre-defined goals for meetings', 'Knowing where to find the information I need for a meeting', 'Understanding my exact role and responsibilities when I'm invited', 'Having tools to manage admin tasks like note-taking or summarization', and 'Having more focus time to sufficiently prepare for meetings'. Each category has an associated bar indicating the level of agreement, measured on a scale from 0% to 100%."}, 
            # {"role": "user", "content": "Provide insightful questions to spark discussion."} 
        ] 
        response = model.chat(
            image=image,
            msgs=messages,
            tokenizer=self.tokenizer
        )
        # print(response)
        # prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # image_list = []
        # if image != None: 
        #     image_list.append(image)
        # else:
        #     image_list = None
        # inputs = processor(prompt, image_list, return_tensors="pt").to(self.device,dtype=self.dtype)
        # generation_args = { 
        #     "max_new_tokens": 500, 
        #     "temperature": 0.0, 
        #     "do_sample": False, 
        # } 

        # # autoregressively complete prompt
        # generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 
        # # response = processor.decode(output[0], skip_special_tokens=True)
        # # prompt = prompt.replace(" <image>"," ")
        # # response = response.replace(prompt,'').strip()
        # generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        # response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

        # del image_list,generate_ids
        # flush()
        
        return response
    
if __name__ == "__main__":
    # input_dir = "F:/ImageSet/pony_caption_output/female/1_0_bikini_char.webp"
    # image_path = "F:/ImageSet/pony_caption_output/female/1_0_bikini_char.webp"
    
    # input_dir = "E:/Development/Bilibili-Image-Grapple/classification/output/bomiao"
    # input_dir = "F:/ImageSet/pony_caption_output/"
    input_dir = "F:/ImageSet/kolors_cosplay/ai_anime/female/aegir_azur_lane"
    # output_dir = "E:/Development/Bilibili-Image-Grapple/classification/output/bomiao_crop_watermark"
    # os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(f"{input_dir}/**", recursive=True)
    image_exts = [".png",".jpg",".jpeg",".webp"]
    image_files = [f for f in files if os.path.splitext(f)[-1].lower() in image_exts]
    
    for image_file in tqdm(image_files):
        print(f"\n{image_file}")
        # image_path = os.path.join(input_dir, image_file)
        
        image = cv2.imread(image_file)
        # image = Image.open(image_path).convert('RGB')
        model = MinicpmModelWrapper()
        result = model.execute(image)
        print(result)
        break