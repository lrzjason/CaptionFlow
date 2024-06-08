

from PIL import Image

# ModelWrappers
from wd14 import WD14ModelWrapper
from idefics2 import Idefics2ModelWrapper
from cogvlm import CogvlmModelWrapper
from kosmos2 import Kosmos2ModelWrapper
from glm4 import Glm4ModelWrapper
from llama3 import Llama3ModelWrapper
from groundingDino import GroundingDinoModelWrapper
from llavaNext import LlavaNextModelWrapper
from ramPlus import RamPlusModelWrapper
from deepseekVL import DeepseekVLModelWrapper

import hpsv2
from textblob import TextBlob

from utils import flush
import os
import time
import json


def save_caption(text_path,caption):
    with open(text_path, 'w', encoding="utf-8") as f:
        f.write(caption)
        f.close()

def read_caption(text_path):
    return open(text_path, encoding='utf-8').read()
# get multiple captions from different models
def proposal(image):
    captions = []
    model_list = [
        {
            "name":"Cogvlm",
            "model":CogvlmModelWrapper
        },
        {
            "name":"DeepseekVL",
            "model":DeepseekVLModelWrapper
        },
    ]
    for model_config in model_list:
        text_path = f"stage_1_proposal.{model_config['name']}"
        if os.path.exists(text_path):
            captions.append({
                "name": model_config['name'],
                "caption": read_caption(text_path)
                })
        else:
            model = model_config['model']()
            # model = model.create()
            # caption = model.execute(model,image).strip()
            caption = model.execute(image).strip()
            # caption = f"{model_config['name']}_caption:\n {result} \n"
            save_caption(text_path,caption)
            captions.append({
                "name": model_config['name'],
                "caption": caption
                })
            # captions.append(caption)
            del model
            flush()
    
    return captions
    
# Step 1, llm reconstruct caption from multiple captions
# Step 2, llm list objects from captions
# Step 3, Object Detection Model check the correctness of the object list
# return Object Detection list and caption
def verification(image,stage1_results):
    llm_model_list = [
        {
            "name":"glm4",
            "model":Glm4ModelWrapper
        },
        # {
        #     "name":"llama3",
        #     "model":Llama3model
        # },
    ]
    
    query = ("I want to use an object detector to check the correctness of an image caption "
             "obtained by an image caption model. Can you help to parse the caption below and "
             "list all nouns that could be detected with an object detection model in the image? "
             "Please only list the object name and ignore the description. "
             "Please use singular for all listed nouns. Caption: {}. "
             "Please concatenate them together with “. ” as separation.")
    
    verification_result_list = []
    for model_config in llm_model_list:
        created_model = False
        caption = ""
        text_path = f"stage_2_verification_step1"
        text_path_step2 = f"stage_2_verification_step2"
        
        if os.path.exists(text_path):
            caption += read_caption(text_path)
        else:
            # collect all stage1_result and mix all of them together
            for stage1_result in stage1_results:
                # stage1_result_caption = stage1_result['caption']
                # stage1_name = f"{stage1_result['name']}.{model_config['name']}"
                if not os.path.exists(text_path) or not os.path.exists(text_path_step2):
                    if created_model == False:
                        model = model_config['model']()
                        # model = model.create()
                        created_model = True
                    
                        caption = model.execute(captions=stage1_result)
                        caption += caption.replace("\n","").strip()
        
            save_caption(text_path,caption)
                
                
        step2_result = ""
        if os.path.exists(text_path_step2):
            step2_result = read_caption(text_path_step2)
        else:
            step2_result = model.execute(query=query,captions=caption)
            step2_result = step2_result.replace("_"," ").replace(",",".").replace("\n","").strip().lower()
            save_caption(text_path_step2,step2_result)
        
        verification_result_list.append({
            "name": model_config['name'],
            "model": model_config['model'],
            # "stage1_name": stage1_name,
            # "stage1_result": stage1_result_caption,
            "stage2_step1_result": caption,
            "stage2_step2_result": step2_result,
            "stage2_step3_result_captions": []
        })
        
        if created_model:
            del model
            flush()
    
    object_detection_list = [
        {
            "name":"groundingDino",
            "model": GroundingDinoModelWrapper
        },
        {
            "name":"wd14",
            "model": WD14ModelWrapper
        },
        {
            "name":"ramPlus",
            "model": RamPlusModelWrapper
        },
    ]
    for od_model_config in object_detection_list:
        created_model = False
        model = None
        model = None
        for verification_result in verification_result_list:
            # if verification_result['stage2_step3_result'] == None:
            #     verification_result['stage2_step3_result'] = {
            #         "name": verification_result['stage1_name'],
            #         "caption_list": []
            #     }
            text_path_step3 = f"stage_2_verification_step3.{od_model_config['name']}"
            if os.path.exists(text_path_step3):
                caption = read_caption(text_path_step3)
            else:
                if created_model == False:
                    model = od_model_config['model']()
                    # model = model.create()
                    created_model = True
                
                caption = model.execute(image=image,query=verification_result['stage2_step2_result'])
                caption = caption.replace(", ",". ").strip()
                save_caption(text_path_step3,caption)
            # verification_result['stage2_step3_result']['caption_list'].append(caption)
            verification_result['stage2_step3_result_captions'].append(caption)
        del model
        flush()
    
    for verification_result in verification_result_list:
        verification_result['stage2_step3_result'] = ". ".join(verification_result['stage2_step3_result_captions'])
    return verification_result_list

def captioning(verification_result_list):
    query = read_caption("prompt_template_captioning.txt")
    
    result = []
    created_model = False
    model = None
    model = None
    for verification_result in verification_result_list:
        # raw caption
        stage2_step1_result = verification_result['stage2_step1_result']
        stage2_step3_result = verification_result['stage2_step3_result']
        refined_text_path = f"stage_3_captioning_step1.{verification_result['name']}"
        refined_caption = ""
        if os.path.exists(refined_text_path): 
            refined_caption = read_caption(refined_text_path)
        else:
            if created_model == False:
                model = verification_result['model']()
                # model = model.create()
                created_model = True
            # object detection result
            formated_query = query.format(stage2_step3_result, stage2_step1_result)
            refined_caption = model.execute(query=formated_query)
            
            refined_caption = refined_caption.replace('```json',"").replace('```',"").replace("\n","")
            save_caption(refined_text_path,refined_caption)
        try:
            json_object = json.loads(refined_caption)
            result.append({
                'name':verification_result['name'],
                'Reasoning':json_object['Reasoning'],
                'Modification':json_object['Modification'],
                'FinalCaption':json_object['FinalCaption'],
            })
        except Exception:
            print("json parse error")
            
    review_query = read_caption("prompt_template_review.txt")
    for caption_result in result:
        reviewed_text_path = f"stage_3_captioning_step2.{verification_result['name']}"
        if os.path.exists(reviewed_text_path): 
            caption_result['ReviewedCaption'] = read_caption(reviewed_text_path)
        else:
            if created_model == False:
                model = verification_result['model']()
                # model = model.create()
                created_model = True
            # object detection result
            formated_query = review_query.format(caption_result['Reasoning'], 
                                                 caption_result['Modification'], 
                                                 caption_result['FinalCaption'])
            reviewed_caption = model.execute(query=formated_query)
            
            save_caption(reviewed_text_path+".raw",reviewed_caption)
            
            begin_str = "—BEGIN Caption: —"
            end_str = "—END Caption—"
            start_index = 0
            if reviewed_caption.index(begin_str) > 0:
                start_index = reviewed_caption.index(begin_str)+len(begin_str)
            end_index = len(reviewed_caption)
            if reviewed_caption.index(end_str) > 0:
                end_index = reviewed_caption.index(end_str)
            reviewed_caption = reviewed_caption[start_index:end_index].replace("\n","").strip()
            save_caption(reviewed_text_path,reviewed_caption)
            caption_result['ReviewedCaption'] = reviewed_caption
    if created_model == True:
        del model
        flush()
    
            
    return result

def evaluation(image,result_list):
    captions = []
    # collect all final output
    for result in result_list:
        # content = result['ReviewedCaption']
        # score = hpsv2.score(image, content, hps_version="v2.1")[0]
        
        # result['ReviewedCaption_hpsv2_score'] = format(score, '.4f')
        captions.append({
            "name": f"CaptionFlow.final.{result['name']}",
            "caption": result['ReviewedCaption']
        })
    # collect all raw vlm output
    model_list = [
        {
            "name":"Cogvlm",
        },
        {
            "name":"DeepseekVL",
        },
    ]
    for model_config in model_list:
        text_path = f"stage_1_proposal.{model_config['name']}"
        captions.append({
                "name": model_config['name'],
                "caption": read_caption(text_path)
                })
        
    # collect all raw vlm output
    for caption in captions:
        content = caption['caption']
        score = hpsv2.score(image, content, hps_version="v2.1")[0]
        
        caption['score'] = format(score, '.4f')
    
    return captions
def main():
    
    start_time = time.time()
    print(f"Script started at {time.ctime(start_time)}")
    
    image_path = "11.jpg"
    image = Image.open(image_path)
    proposal_results = proposal(image)
    verification_result_list = verification(image,proposal_results)
    captioning_result = captioning(verification_result_list)
    result = evaluation(image,captioning_result)
    # print(result)
    print(json.dumps(result, indent=4))
    end_time = time.time()  # get the end time

    execution_time = end_time - start_time  # calculate the execution time

    # log the end of the script
    print(f"Script ended at {time.ctime(end_time)}")
    print(f"Total execution time: {execution_time} seconds")

if __name__ == '__main__':
    main()