import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from ModelWrapper import ModelWrapper
from utils import flush,get_device


class GroundingDinoModelWrapper(ModelWrapper):
    def __init__(self,device=None):
        super().__init__()
        self.device = get_device(device)
        self.model_repo_id = "IDEA-Research/grounding-dino-base"
        self.processor = AutoProcessor.from_pretrained(self.model_repo_id)

    def create(self):
        model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_repo_id).to(self.device)
        return model
    
    def execute(self,model,image=None,query=""):
        processor = self.processor
        inputs = processor(images=image, text=query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )
        labels = ". ".join(results[0]['labels'])
        return labels
    

if __name__ == "__main__":
    image_path = "2.webp"
    image = Image.open(image_path)
    groundingDino = GroundingDinoModelWrapper()
    groundingDino_model = groundingDino.create()
    result = groundingDino.execute(groundingDino_model,image)
    print(result)