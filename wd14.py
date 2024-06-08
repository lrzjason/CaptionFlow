import os
import csv
import torch
import numpy as np
import pandas as pd
import onnxruntime
from PIL import Image
import cv2
from pathlib import Path
from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import re
import gc

from ModelWrapper import ModelWrapper
from utils import flush

def download_model_files(model_repo_id):
    # Define local paths to save the files
    local_model_path = hf_hub_download(repo_id=model_repo_id, filename='model.onnx')
    local_tags_path = hf_hub_download(repo_id=model_repo_id, filename='selected_tags.csv')

    return local_model_path, local_tags_path

def preprocess_image(image):
    image = image.convert('RGBA')
    bg = Image.new('RGBA', image.size, 'WHITE')
    bg.paste(image, mask=image)
    image = bg.convert('RGB')
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # convert to BGR format
    h, w = image.shape[:2]
    size = max(h, w)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    image = np.pad(image, [(pad_h, pad_h), (pad_w, pad_w), (0, 0)], mode='constant', constant_values=255)
    image = cv2.resize(image, (448, 448), interpolation=cv2.INTER_AREA)
    image = np.expand_dims(image, 0)
    return image.astype(np.float32)


class WD14ModelWrapper(ModelWrapper):
    def __init__(self):
        super().__init__()
        self.model_repo_id = 'SmilingWolf/wd-swinv2-tagger-v3'
        model_path, tags_path = download_model_files(self.model_repo_id)
        self.model_path = model_path
        self.tags_path = tags_path
        self.tag_only = True
        self.character_category = 4
        self.model = onnxruntime.InferenceSession(self.model_path, providers=['CUDAExecutionProvider'])
    def execute(self,image=None,query=None,filter_tags=['1girl','solo','questionable','general','sensitive'], tag_threshold=0.4):
        model = self.model
        tags_scores = []
        processed_image = preprocess_image(image)
        result = model.run(None, {model.get_inputs()[0].name: processed_image})[0]
        tags = pd.read_csv(self.tags_path)
        tags.reset_index(inplace=True)
        result_df = pd.DataFrame(result[0], columns=['Score'])
        result_with_tags = pd.concat([tags, result_df], axis=1)
        tags_filtered = result_with_tags[['name', 'Score', 'category']]
        tags_filtered = tags_filtered[~tags_filtered['name'].isin(filter_tags)]
        tags_scores.append(tags_filtered.set_index('name'))
        averaged_tags_scores = tags_scores[0].reset_index()

        averaged_tags_scores.columns = ['name', 'Score', 'category']  # rename columns
        averaged_tags_scores = averaged_tags_scores[averaged_tags_scores['Score'] > tag_threshold]
        averaged_tags_scores.sort_values('Score', ascending=False, inplace=True)
        # print(averaged_tags_scores)
        
        tag_string = ""
        for _, row in averaged_tags_scores.iterrows():
            if self.tag_only:
                tag = row['name'].replace('_',' ')
                if tag == '1girl':
                    tag = 'woman'
                if tag == '1boy':
                    tag = 'man'
                tag_string += f"{tag}, "
            else:
                if row['category'] == self.character_category:
                    tag_string += f"[characeter_{row['name']}: {row['Score']:.2f}], "
                else:
                    tag_string += f"[{row['name']}: {row['Score']:.2f}], "
                    
        
        # clear memory
        del averaged_tags_scores,tags_scores,tags_filtered,result_with_tags,result_df,tags,result,processed_image
        flush()
        
        # print(tag_string)
        return tag_string




                
