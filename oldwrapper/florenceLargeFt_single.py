
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
import numpy as np
from Pylette import extract_colors

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

def sort_bboxes_by_priority(image, bboxes, labels):
    """
    Sort bounding boxes based on proximity to image center and size.

    Args:
        image (numpy.ndarray): The input image (height, width, channels).
        bboxes (list): List of bounding boxes, where each bbox is [x1, y1, x2, y2].

    Returns:
        list: Sorted list of bounding boxes.
    """
    # Get the image center
    img_height, img_width = image.shape[:2]
    img_center = (img_width / 2, img_height / 2)
    
        # Get the image center
    img_height, img_width = image.shape[:2]
    img_center = (img_width / 2, img_height / 2)
    
    # Calculate scores for each bbox
    scores = []
    for bbox,label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox
        # Calculate bbox center
        bbox_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        # Calculate distance to image center
        distance_to_center = np.sqrt((bbox_center[0] - img_center[0]) ** 2 +
                                     (bbox_center[1] - img_center[1]) ** 2)
        # Calculate bbox size (area)
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_size = bbox_width * bbox_height
        # Calculate score
        score = bbox_size / (1 + distance_to_center)
        # for readability reverse - value to + value
        # score = -1 * score
        scores.append((score, bbox, label))
    
    # Sort by score
    sorted_bboxes = [(score, bbox, label) for score, bbox, label in sorted(scores, key=lambda x: x[0])]
    
    return sorted_bboxes


if __name__ == "__main__":
    # image_path = "F:/ImageSet/sd3_test/1_creative_photo/ComfyUI_temp_zpsmu_00236_.png"
    # image = Image.open(image_path)
    
    # input_dir = "E:/Development/Bilibili-Image-Grapple/classification/output/bomiao"
    # input_dir = "F:/ImageSet/pony_caption_output/"
    # # output_dir = "E:/Development/Bilibili-Image-Grapple/classification/output/bomiao_crop_watermark"
    # # os.makedirs(output_dir, exist_ok=True)
    # files = glob.glob(f"{input_dir}/**", recursive=True)
    # image_exts = [".png",".jpg",".jpeg",".webp"]
    # image_files = [f for f in files if os.path.splitext(f)[-1].lower() in image_exts]
    # image_files = ["E:/Development/Bilibili-Image-Grapple/classification/output/maileji - Copy/maileji_3.png"]
    # print(image_files)
    model = FlorenceLargeFtModelWrapper()
    image = cv2.imread("E:/Media/0AIpainting/20241205/test.png")
    result = model.execute(image,other_prompt="<OD>")
    # {'bboxes': [[0.32899999618530273, 553.2135009765625, 134.56100463867188, 613.4204711914062], [286.5589904785156, 261.0614929199219, 354.9909973144531, 309.42449951171875], [571.4730224609375, 883.8584594726562, 657.0130004882812, 915.4425048828125], [0.32899999618530273, 552.2265014648438, 657.0130004882812, 985.5194702148438], [370.7829895019531, 99.19349670410156, 657.0130004882812, 596.6414794921875], [0.32899999618530273, 727.9124755859375, 186.54299926757812, 840.4304809570312], [443.82098388671875, 780.2235107421875, 657.0130004882812, 880.8974609375], [0.32899999618530273, 729.886474609375, 153.64300537109375, 823.6514892578125], [68.10299682617188, 640.0695190429688, 145.7469940185547, 680.5364990234375], [525.4129638671875, 629.2124633789062, 657.0130004882812, 785.1585083007812]], 'labels': ['bowl', 'coffee cup', 'fork', 'kitchen & dining room table', 'person', 'plate', 'plate', 'plate', 'plate', 'tableware']}
    
    # image = "./test.webp"
    palette = extract_colors(image=image, palette_size=10)

    # Display the palette, and save the image to file
    palette.display(save_to_file=True,filename=f"0_test_palette")
    sorted_bboxes = sort_bboxes_by_priority(image, result['bboxes'], result['labels'])
    print(sorted_bboxes)
    for score, bbox, label in sorted_bboxes:
        # print(scored_bbox)
        # score = scored_bbox[0]
        # bbox = scored_bbox[1]
        # label = scored_bbox[2]
        x1, y1, x2, y2 = map(int, bbox)  # Convert coordinates to integers
        # Draw rectangle
        
        # get bbox area image
        bbox_area_image = image[y1:y2, x1:x2]
        # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box with thickness 2
        # # Put label text
        # cv2.putText(
        #     image, f"{label}:{score}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        # )  # Green text
        # image = "./test.webp"
        
        # save bbox_area_image
        cv2.imwrite(f"0_{label}_{score}.webp", bbox_area_image)
        
        palette = extract_colors(image=bbox_area_image, palette_size=10, mode="MC", sort_mode="frequency")
        # Access colors by index
        # most_common_color = palette[0]
        # least_common_color = palette[-1]

        # # Get color information
        # print(most_common_color.rgb)
        # print(most_common_color.hls)
        # print(most_common_color.hsv)

        # Display the palette, and save the image to file
        palette.display(save_to_file=True,filename=f"0_{label}_{score}")

    # Display the image using OpenCV
    cv2.imshow("Image with Bounding Boxes", image)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    
    
    
    # input_dir = "F:/ImageSet/niji"
    # loop input_dir for each image
    # for image_file in tqdm(image_files):
    #     text_file = os.path.splitext(image_file)[0] + ".txt"
    #     # image_path = os.path.join(input_dir, image_file)
        
    #     image = cv2.imread(image_file)
    #     # get webp params
    #     # filesize = os.path.getsize(image_file) 
    #     # # print('File: ' + file + ' Size: ' + str(filesize) + ' bytes')
    #     # filesize_mb = filesize / 1024 / 1024
    #     # # skip low filesize images
    #     # if filesize_mb < 0.5:
    #     #     print("skip low filesize image: ", image_file)
    #     #     continue
    #     # lossless, quality = get_webp_params(filesize_mb)
        
    #     # image = cv2.resize(image, (int(image.shape[1]*0.7), int(image.shape[0]*0.7)), interpolation=cv2.INTER_AREA)
    #     # ori_image = image.copy()
    #     # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     # image = Image.open(image_path).convert('RGB')
    #     result = model.execute(image)
        
    #     # read text file
    #     with open(text_file, "r", encoding="utf-8") as f:
    #         text = f.read()
    #         new_content = "二次元动漫风格, anime artwork, " + result + ", " + text
    #         # rename original text file to _ori.txt
    #         old_text_file = text_file.replace(".txt","_ori.txt")
    #         if os.path.exists(old_text_file):
    #             continue
    #         # save new content to text file
    #         with open(old_text_file, "w", encoding="utf-8") as ori_f:
    #             ori_f.write(text)
    #             print("save ori content to text file: ", old_text_file)
    #         # save new content to text file
    #         with open(text_file, "w", encoding="utf-8") as new_f:
    #             new_f.write(new_content)
    #             print("save new content to text file: ", text_file)
            
        
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