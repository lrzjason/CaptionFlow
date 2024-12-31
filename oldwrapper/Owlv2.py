import torch
from PIL import Image
from pathlib import Path
from huggingface_hub import hf_hub_download

from transformers import AutoProcessor, Owlv2ForObjectDetection
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from ModelWrapper import ModelWrapper

import numpy as np


class Owlv2ModelWrapper(ModelWrapper):
    def __init__(self):
        super().__init__()
        self.model_repo_id = "google/owlv2-base-patch16-ensemble"
        self.processor = AutoProcessor.from_pretrained(self.model_repo_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(self.model_repo_id)
        self.tag_threshold = 0.4

    
    @torch.no_grad()
    def execute(self,image=None,query=None):
        # tag_string, scores = self.predict(image)
        processor = self.processor
        model = self.model
        
        # split query str to texts array
        texts = []
        if query is not None:
            texts = [query.split(".")]
        inputs = processor(text=texts, images=image, return_tensors="pt",padding=True,truncation=True)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # Note: boxes need to be visualized on the padded, unnormalized image
        # hence we'll set the target image sizes (height, width) based on that

        def get_preprocessed_image(pixel_values):
            pixel_values = pixel_values.squeeze().numpy()
            unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
            unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
            unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
            unnormalized_image = Image.fromarray(unnormalized_image)
            return unnormalized_image

        unnormalized_image = get_preprocessed_image(inputs.pixel_values)

        target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to final bounding boxes and scores
        results = processor.post_process_object_detection(
            outputs=outputs, threshold=0.2, target_sizes=target_sizes
        )

        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        text = texts[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        result = []
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            score = round(score.item(), 3)
            # print(f"Detected {text[label]} with confidence {score} at location {box}")
            tag = text[label].strip()
            if not tag in result:
                result.append(tag)
            
        return ". ".join(result)

if __name__ == "__main__":
    image_path = "1.png"
    image = Image.open(image_path).convert("RGB")
    joyTag = Owlv2ModelWrapper()
    result = joyTag.execute(image,query="woman. long hair. looking at viewer. skirt. brown hair. black hair. photoshop (medium). man. ribbon. long sleeves. original. brown eyes. very long hair. standing. hair ribbon. weapon. ponytail. japanese clothes. parted lips. outdoors. day. sword. wide sleeves. kimono. blurry. lips. red ribbon. sash. profile. leaf. blurry background. katana. sheath. hakama. hakama skirt. tassel. sheathed. red lips. blue hakama. hanfu")
    print("\nresult\n")
    print(result)