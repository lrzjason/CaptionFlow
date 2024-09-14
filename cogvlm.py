"""
This is a demo for using CogAgent and CogVLM in CLI
Make sure you have installed vicuna-7b-v1.5 tokenizer model (https://huggingface.co/lmsys/vicuna-7b-v1.5), full checkpoint of vicuna-7b-v1.5 LLM is not required.
In this demo, We us chat template, you can use others to replace such as 'vqa'.
Strongly suggest to use GPU with bfloat16 support, otherwise, it will be slow.
Mention that only one picture can be processed at one conversation, which means you can not replace or insert another picture during the conversation.
"""

import argparse
import torch

from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer, BitsAndBytesConfig
import os
import time
from ModelWrapper import ModelWrapper
from utils import flush,get_device
import glob
from tqdm import tqdm
import cv2

class CogvlmModelWrapper(ModelWrapper):

    def __init__(self,device=None,dtype=None,tokenizer_repo_id="lmsys/vicuna-7b-v1.5"):
        super().__init__()
        self.device = get_device(device)
        self.model_repo_id = "THUDM/cogvlm-chat-hf"
        self.tokenizer_repo_id = tokenizer_repo_id
        if dtype == None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.tokenizer = LlamaTokenizer.from_pretrained(self.tokenizer_repo_id,padding_side="right")
        
        self.gen_kwargs = {
            'min_new_tokens':100,
            'max_new_tokens':350,
            'num_beams':1,
            'length_penalty':1,
            'top_k':60,
            'top_p':0.6,
            'repetition_penalty': 1.15,
            'no_repeat_ngram_size':0,
            "do_sample": True,
            "temperature": 0.6,
        } 
        self.prompt = f'Describe the image precisely, detailing every element, interaction and background. Include composition, angle and perspective. Use only facts and concise language; avoid interpretations or speculation:'
        self.starts_with = f'The image showcases '
        
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self.dtype,
            bnb_4bit_quant_type="fp4",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_repo_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            # load_in_4bit=True,
            # bnb_4bit_compute_dtype=self.dtype
            quantization_config=quantization_config,
            trust_remote_code=True
        ).eval()
        # .to(self.device)
        
        
    def execute(self, image=None,prompt=None,starts_with=None):
        model = self.model
        if prompt != None:
            self.prompt = prompt
        if starts_with != None:
            self.starts_with = starts_with
        
        torch_type = self.dtype
        query = f'Question: {self.prompt} Answer: {self.starts_with}'
        history = []
        input_by_model = model.build_conversation_input_ids(self.tokenizer, query=query, history=history, images=[image])

        prepare_images = []
        if self.gen_kwargs['num_beams'] > 1:
            prepare_images = [[input_by_model['images'][0].to(self.device).to(torch_type)] for _ in range(self.gen_kwargs['num_beams'])]
        else:
            prepare_images = [[input_by_model['images'][0].to(self.device).to(torch_type)]] if image is not None else None
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(self.device),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(self.device),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(self.device),
            'images': prepare_images,
        }
        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [[input_by_model['cross_images'][0].to(self.device).to(torch_type)]]

        response = ""
        with torch.no_grad():
            outputs = model.generate(**inputs, **self.gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0])
            response = response.split("</s>")[0]
            response = replace(response)
        
        # clear memory
        del outputs,inputs,prepare_images,input_by_model
        flush()
        
        return response
def replace(response):
    # trancate hallucination 
    if "Answer:" in response:
        response = response[:response.index("Answer:")]

    if "watermark" in response:
        start = response.find("watermark")
        sentence_start = response.rfind('.', 0, start) + 1
        response = response[:sentence_start]

    if "caption " in response:
        start = response.find("caption ")
        sentence_start = response.rfind('.', 0, start) + 1
        response = response[:sentence_start]

    if "signature " in response:
        start = response.find("signature ")
        sentence_start = response.rfind('.', 0, start) + 1
        response = response[:sentence_start]

    if "signed by " in response:
        start = response.find("signed by ")
        sentence_start = response.rfind('.', 0, start) + 1
        response = response[:sentence_start]

    if "bottom right corner" in response:
        start = response.find("bottom right corner")
        senteglobnce_start = response.rfind('.', 0, start) + 1
        response = response[:sentence_start]
    return response


if __name__ == "__main__":
    # image_path = "F:/ImageSet/sd3_test/1_creative_photo/ComfyUI_temp_zpsmu_00236_.png"
    # image = Image.open(image_path)
    
    # input_dir = "E:/Development/Bilibili-Image-Grapple/classification/output/bomiao"
    input_dir = "F:/ImageSet/kolors_cosplay/ai_anime/female/aegir_azur_lane"
    # output_dir = "E:/Development/Bilibili-Image-Grapple/classification/output/bomiao_crop_watermark"
    # os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(f"{input_dir}/**", recursive=True)
    image_exts = [".png",".jpg",".jpeg",".webp"]
    image_files = [f for f in files if os.path.splitext(f)[-1].lower() in image_exts]
    # image_files = ["E:/Development/Bilibili-Image-Grapple/classification/output/maileji - Copy/maileji_3.png"]
    # print(image_files)
    model = CogvlmModelWrapper()
    # input_dir = "F:/ImageSet/niji"
    # loop input_dir for each image
    for image_file in tqdm(image_files):
        text_file = os.path.splitext(image_file)[0] + ".txt"
        # image_path = os.path.join(input_dir, image_file)
        
        image = cv2.imread(image_file)
        # get webp params
        # filesize = os.path.getsize(image_file) 
        # # print('File: ' + file + ' Size: ' + str(filesize) + ' bytes')
        # filesize_mb = filesize / 1024 / 1024
        # # skip low filesize images
        # if filesize_mb < 0.5:
        #     print("skip low filesize image: ", image_file)
        #     continue
        # lossless, quality = get_webp_params(filesize_mb)
        
        # image = cv2.resize(image, (int(image.shape[1]*0.7), int(image.shape[0]*0.7)), interpolation=cv2.INTER_AREA)
        # ori_image = image.copy()
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = Image.open(image_path).convert('RGB')
        result = model.execute(image)
        print(result)
        
        # read text file
        # with open(text_file, "r", encoding="utf-8") as f:
        #     text = f.read()
        #     new_content = "二次元动漫风格, anime artwork, " + result + ", " + text
        #     # rename original text file to _ori.txt
        #     old_text_file = text_file.replace(".txt","_ori.txt")
        #     if os.path.exists(old_text_file):
        #         continue
        #     # save new content to text file
        #     with open(old_text_file, "w", encoding="utf-8") as ori_f:
        #         ori_f.write(text)
        #         print("save ori content to text file: ", old_text_file)
        #     # save new content to text file
        #     with open(text_file, "w", encoding="utf-8") as new_f:
        #         new_f.write(new_content)
        #         print("save new content to text file: ", text_file)
            
        
        # ############# OCR for watermark ################
        # break
        # result = model.execute(image,other_prompt="<OCR_WITH_REGION>")
        
        # crop_image = False
        # quad_boxes = result["quad_boxes"]
        # for i, quad_box in enumerate(quad_boxes):
        #     x1,y1,x2,y2,x3,y3,x4,y4 = quad_box
            
        #     # only handle fixed bottom region
        #     if y1 > 0.8*image.shape[0]:
        #         cv2.line(image, (0, int(y1)), (image.shape[1], int(y1)), (0, 0, 255), 1)
        #         # crop image
        #         crop_img = image[0:int(y1), :]
        #         crop_image = True
        # # show image for debug
        # # cv2.imshow('Image', image)
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()
        
        # # save cropped image to output_dir
        # file_name, file_ext = os.path.splitext(os.path.basename(image_file))
        # output_path = os.path.join(output_dir, f"{file_name}.webp")
        
        # print("save image: ", output_path)
        # if crop_image:
        #     cv2.imwrite(output_path, crop_img, [int(cv2.IMWRITE_WEBP_QUALITY), quality])
        # else:
        #     cv2.imwrite(output_path, image, [int(cv2.IMWRITE_WEBP_QUALITY), quality])
        break