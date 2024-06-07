

from PIL import Image
from wd14 import WD14ModelWrapper
from idefics2 import Idefics2ModelWrapper
from cogvlm import CogvlmModelWrapper
from kosmos2 import Kosmos2ModelWrapper
from glm4 import Glm4ModelWrapper
from llama3 import Llama3ModelWrapper
from groundingDino import GroundingDinoModelWrapper
from llavaNext import LlavaNextModelWrapper
from ramPlus import RamPlusModelWrapper
from utils import flush
import os
import time

def save_caption(text_path,caption):
    with open(text_path, 'w', encoding="utf-8") as f:
        f.write(caption)
        f.close()

def read_caption(text_path):
    return open(text_path, encoding='utf-8').read()
# get multiple captions from different models
def proposal(image):
    captions = ""
    model_list = [
        {
            "name":"cogvlm",
            "modelWrapper":CogvlmModelWrapper
        },
        {
            "name":"idefics2",
            "modelWrapper":Idefics2ModelWrapper
        },
        {
            "name":"llavaNext",
            "modelWrapper":LlavaNextModelWrapper
        },
        {
            "name":"kosmos2",
            "modelWrapper":Kosmos2ModelWrapper
        },
    ]
    for model_config in model_list:
        text_path = f"result_1_caption.{model_config['name']}"
        if os.path.exists(text_path):
            captions += read_caption(text_path)
        else:
            modelWrapper = model_config['modelWrapper']()
            model = modelWrapper.create()
            caption = modelWrapper.execute(model,image).strip()
            # caption = f"{model_config['name']}_caption:\n {result} \n"
            save_caption(text_path,caption)
            captions += caption
            del modelWrapper,model
            flush()
    
    return captions
    
# Step 1, llm reconstruct caption from multiple captions
# Step 2, llm list objects from captions
# Step 3, Object Detection Model check the correctness of the object list
# return Object Detection list and caption
def verification(image,proposal_result):
    llm_model_list = [
        {
            "name":"glm4",
            "modelWrapper":Glm4ModelWrapper
        },
        # {
        #     "name":"llama3",
        #     "modelWrapper":Llama3ModelWrapper
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
        text_path = f"result_2_verification_step1.{model_config['name']}"
        text_path_step2 = f"result_2_verification_step2.{model_config['name']}"
        created_model = False
        if not os.path.exists(text_path) or not os.path.exists(text_path_step2):
            modelWrapper = model_config['modelWrapper']()
            model = modelWrapper.create()
            created_model = True
            
        caption = ""
        if os.path.exists(text_path):
            caption = read_caption(text_path)
        else:
            caption = modelWrapper.execute(model,captions=proposal_result)
            caption = caption.replace("\n","").strip()
            save_caption(text_path,caption)
            
            
        step2_result = ""
        if not os.path.exists(text_path_step2):
            step2_result = modelWrapper.execute(model,query=query,captions=caption)
            step2_result = step2_result.replace("_"," ").replace(",",".").replace("\n","").strip().lower()
            save_caption(text_path_step2,step2_result)
        else:
            step2_result = read_caption(text_path_step2)
        
        verification_result_list.append({
            "name": model_config['name'],
            "modelWrapper": model_config['modelWrapper'],
            "proposal_result": proposal_result,
            "step1_result": caption,
            "step2_result": step2_result,
            "step3_result": []
        })
        
        if created_model:
            del modelWrapper,model
            flush()
    
    
    ojbect_detection_list = [
        {
            "name":"groundingDino",
            "modelWrapper": GroundingDinoModelWrapper
        },
        {
            "name":"wd14",
            "modelWrapper": WD14ModelWrapper
        },
        {
            "name":"ramPlus",
            "modelWrapper": RamPlusModelWrapper
        },
    ]
    for verification_result in verification_result_list:
        for od_model_config in ojbect_detection_list:
            text_path_step3 = f"result_2_verification_step3.{od_model_config['name']}.{verification_result['name']}"
            if os.path.exists(text_path_step3):
                caption = read_caption(text_path_step3)
            else:
                modelWrapper = od_model_config['modelWrapper']()
                model = modelWrapper.create()
                caption = modelWrapper.execute(model,image=image,query=verification_result['step2_result'])
                caption = caption.replace(", ",". ").strip()
                save_caption(text_path_step3,caption)
                del modelWrapper,model
                flush()
            verification_result['step3_result'].append({
                "result": caption,
                "name": od_model_config['name']
            })
    return verification_result_list

def captioning(verification_result_list):
    query = read_caption("captioning_prompt_template.txt")
    
    for verification_result in verification_result_list:
        modelWrapper = verification_result['modelWrapper']()
        model = modelWrapper.create()
        # raw caption
        step1_result = verification_result['step1_result']
        for step3_result in verification_result['step3_result']:
            refined_text_path = f"result_3_captioning.{verification_result['name']}.{step3_result['name']}"
            
            if os.path.exists(refined_text_path): continue
            # object detection result
            formated_query = query.format(step3_result['result'], step1_result)
            refined_caption = modelWrapper.execute(model,query=formated_query)
            
            save_caption(refined_text_path,refined_caption)
        del modelWrapper,model
        flush()
    
def main():
    
    start_time = time.time()
    print(f"Script started at {time.ctime(start_time)}")
    
    image_path = "4.png"
    image = Image.open(image_path)
    proposal_result = proposal(image)
    verification_result_list = verification(image,proposal_result)
    refined_caption = captioning(verification_result_list)
    
    end_time = time.time()  # get the end time

    execution_time = end_time - start_time  # calculate the execution time

    # log the end of the script
    print(f"Script ended at {time.ctime(end_time)}")
    print(f"Total execution time: {execution_time} seconds")

if __name__ == '__main__':
    main()