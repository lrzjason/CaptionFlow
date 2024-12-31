
from ModelWrapper import ModelWrapper
from utils import flush,get_device

# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoProcessor, AutoModelForCausalLM 
import torch

from PIL import Image, ImageDraw
import os
import glob
import cv2
from tqdm import tqdm

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
    
    def execute(self,image=None, other_prompt=""):
        model = self.model
        processor = self.processor
        # image = Image.open(requests.get(url, stream=True).raw)
        prompt = self.prompt
        if other_prompt != "":
            prompt = other_prompt
            crop_response = False

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
            # do_sample=False,
            num_beams=3
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        response = processor.post_process_generation(generated_text, task=prompt, image_size=(image.shape[1], image.shape[0]))
        response = response[prompt]
        
        return response
    
# based on file size, return lossless, quality
def get_webp_params(filesize_mb):
    if filesize_mb <= 2:
        return (False, 100)
    if filesize_mb <= 4:
        return (False, 90)
    return (False, 80)


if __name__ == "__main__":
    # image_path = "F:/ImageSet/sd3_test/1_creative_photo/ComfyUI_temp_zpsmu_00236_.png"
    # image = Image.open(image_path)
    
    # input_dir = "E:/Development/Bilibili-Image-Grapple/classification/output/bomiao"
    input_dir = "F:/ImageSet/pony_caption_output/"
    # output_dir = "E:/Development/Bilibili-Image-Grapple/classification/output/bomiao_crop_watermark"
    # os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(f"{input_dir}/**", recursive=True)
    image_exts = [".png",".jpg",".jpeg",".webp"]
    image_files = [f for f in files if os.path.splitext(f)[-1].lower() in image_exts]
    # image_files = ["E:/Development/Bilibili-Image-Grapple/classification/output/maileji - Copy/maileji_3.png"]
    # print(image_files)
    model = FlorenceLargeFtModelWrapper()
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
        
        # read text file
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()
            new_content = "二次元动漫风格, anime artwork, " + result + ", " + text
            # rename original text file to _ori.txt
            old_text_file = text_file.replace(".txt","_ori.txt")
            if os.path.exists(old_text_file):
                continue
            # save new content to text file
            with open(old_text_file, "w", encoding="utf-8") as ori_f:
                ori_f.write(text)
                print("save ori content to text file: ", old_text_file)
            # save new content to text file
            with open(text_file, "w", encoding="utf-8") as new_f:
                new_f.write(new_content)
                print("save new content to text file: ", text_file)
            
        
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
        # # break