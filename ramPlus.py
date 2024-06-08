'''
 * The Recognize Anything Plus Model (RAM++)
 * Written by Xinyu Huang
'''
# import argparse
import numpy as np
import random

import torch

from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform


from ModelWrapper import ModelWrapper
from utils import flush,get_device
from ram.utils import build_openset_llm_label_embedding
import json
import torch.nn as nn


class RamPlusModelWrapper(ModelWrapper):
    def __init__(self,device=None,dtype=None,openset=True):
        super().__init__()
        self.device = get_device(device)
        if dtype == None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.image_size = 384
        self.pretrained = "ram/weight/ram_plus_swin_large_14m.pth"
        
        
        self.llm_tag_des = "ram/openimages_rare_200_llm_tag_descriptions.json"
        self.openset = openset
        
        
        print('Building tag embedding:')
        with open(self.llm_tag_des, 'rb') as fo:
            llm_tag_des = json.load(fo)
        openset_label_embedding, openset_categories = build_openset_llm_label_embedding(llm_tag_des)
        self.openset_label_embedding = openset_label_embedding
        self.openset_categories = openset_categories

        self.openset_model = None
        model = ram_plus(pretrained=self.pretrained,
                            image_size=self.image_size,
                            vit='swin_l')
            
        self.model = model.eval().to(self.device)
        
        self.openset_model = ram_plus(pretrained=self.pretrained,
                                image_size=self.image_size,
                                vit='swin_l')
        self.openset_model.tag_list = np.array(self.openset_categories)
        
        self.openset_model.label_embed = nn.Parameter(self.openset_label_embedding.float())

        self.openset_model.num_class = len(self.openset_categories)
        # the threshold for unseen categories is often lower
        self.openset_model.class_threshold = torch.ones(self.openset_model.num_class) * 0.5
        
        self.openset_model = self.openset_model.eval().to(self.device)
            
    def execute(self,image=None,query=None):
        model = self.model
        device = self.device
        transform = get_transform(image_size=self.image_size)
        image = transform(image).unsqueeze(0).to(device)
        res = inference(image, model)
        result = res[0].replace(" | ",". ")
        
        openset_res = inference(image, self.openset_model)
        openset_result = openset_res[0].replace(" | ",". ").lower()
        result = result + ". " + openset_result
        return result
    
if __name__ == "__main__":
    image_path = "2.webp"
    image = Image.open(image_path)
    ramPlus = RamPlusModelWrapper()
    result = ramPlus.execute(image)
    print(result)
