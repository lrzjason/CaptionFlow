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

import torchvision.transforms.functional as TVF

from ModelWrapper import ModelWrapper
from utils import flush

from joytag.Models import VisionModel

def download_model_files(model_repo_id):
    # Define local paths to save the files
    local_model_path = hf_hub_download(repo_id=model_repo_id,filename='model.safetensors')
    local_tags_path = hf_hub_download(repo_id=model_repo_id, filename='top_tags.txt')
    local_config_path = hf_hub_download(repo_id=model_repo_id,filename='config.json')
    model_path = Path(local_model_path).parent.absolute()

    return model_path,local_tags_path

def prepare_image(image: Image.Image, target_size: int) -> torch.Tensor:
	# Pad image to square
	image_shape = image.size
	max_dim = max(image_shape)
	pad_left = (max_dim - image_shape[0]) // 2
	pad_top = (max_dim - image_shape[1]) // 2

	padded_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
	padded_image.paste(image, (pad_left, pad_top))

	# Resize image
	if max_dim != target_size:
		padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)
	
	# Convert to tensor
	image_tensor = TVF.pil_to_tensor(padded_image) / 255.0

	# Normalize
	image_tensor = TVF.normalize(image_tensor, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

	return image_tensor


class JoyTagModelWrapper(ModelWrapper):
    def __init__(self):
        super().__init__()
        self.model_repo_id = 'fancyfeast/joytag'
        model_path,tags_path = download_model_files(self.model_repo_id)
        self.model_path = model_path
        self.tags_path = tags_path
        self.top_tags = []
        with open(Path(self.tags_path), 'r') as f:
            self.top_tags = [line.strip() for line in f.readlines() if line.strip()]

        self.tag_only = True
        self.character_category = 4
        
        self.model = VisionModel.load_model(model_path)
        self.model = self.model.eval().to('cuda')
        self.tag_threshold = 0.4
    
    @torch.no_grad()
    def execute(self,image=None,query=None,filter_tags=['1girl','solo','questionable','general','sensitive'], tag_threshold=0.7):
        # tag_string, scores = self.predict(image)

        model = self.model
        top_tags = self.top_tags
        tag_threshold = self.tag_threshold
        
        image_tensor = prepare_image(image, model.image_size)
        batch = {
            'image': image_tensor.unsqueeze(0).to('cuda'),
        }

        with torch.amp.autocast_mode.autocast('cuda', enabled=True):
            preds = model(batch)
            tag_preds = preds['tags'].sigmoid().cpu()
        
        scores = {top_tags[i]: tag_preds[0][i] for i in range(len(top_tags))}
        predicted_tags = [tag for tag, score in scores.items() if score > tag_threshold]
        tag_string = ""
        for tag_item in predicted_tags:
            tag = tag_item.replace('_',' ')
            if tag == '1girl':
                tag = 'woman'
            if tag == '1boy':
                tag = 'man'
            tag_string += f"{tag}, "
        # tag_string = ', '.join(predicted_tags)

        return tag_string

if __name__ == "__main__":
    image_path = "1.png"
    image = Image.open(image_path)
    joyTag = JoyTagModelWrapper()
    result = joyTag.execute(image)
    print(result)