from transformers import VideoMAEImageProcessor, VideoMAEModel, VideoMAEConfig, PreTrainedModel

import torch.nn.functional as F
import torch.nn as nn
import torch
from PIL import Image
from torchvision import transforms

class CustomVideoMAEConfig(VideoMAEConfig):
    def __init__(self, scale_label2id=None, scale_id2label=None, movement_label2id=None, movement_id2label=None, **kwargs):
        super().__init__(**kwargs)
        self.scale_label2id = scale_label2id if scale_label2id is not None else {}
        self.scale_id2label = scale_id2label if scale_id2label is not None else {}
        self.movement_label2id = movement_label2id if movement_label2id is not None else {}
        self.movement_id2label = movement_id2label if movement_id2label is not None else {}


class CustomModel(PreTrainedModel):
    config_class = CustomVideoMAEConfig

    def __init__(self, config, model_name, scale_num_classes, movement_num_classes):
        super().__init__(config)
        self.vmae = VideoMAEModel.from_pretrained(model_name, ignore_mismatched_sizes=True)
        self.fc_norm = nn.LayerNorm(config.hidden_size) if config.use_mean_pooling else None
        self.scale_cf = nn.Linear(config.hidden_size, scale_num_classes)
        self.movement_cf = nn.Linear(config.hidden_size, movement_num_classes)

    def forward(self, pixel_values, scale_labels=None, movement_labels=None):

        vmae_outputs = self.vmae(pixel_values)
        sequence_output = vmae_outputs[0]

        if self.fc_norm is not None:
            sequence_output = self.fc_norm(sequence_output.mean(1))
        else:
            sequence_output = sequence_output[:, 0]

        scale_logits = self.scale_cf(sequence_output)
        movement_logits = self.movement_cf(sequence_output)

        if scale_labels is not None and movement_labels is not None:
            loss = F.cross_entropy(scale_logits, scale_labels) + F.cross_entropy(movement_logits, movement_labels)
            return {"loss": loss, "scale_logits": scale_logits, "movement_logits": movement_logits}
        return {"scale_logits": scale_logits, "movement_logits": movement_logits}


scale_lab2id = {"ECS": 0, "CS": 1, "MS": 2, "FS": 3, "LS": 4}
scale_id2lab = {v:k for k,v in scale_lab2id.items()}
movement_lab2id = {"Static": 0, "Motion": 1, "Pull": 2, "Push": 3}
movement_id2lab = {v:k for k,v in movement_lab2id.items()}

config = CustomVideoMAEConfig(scale_lab2id, scale_id2lab, movement_lab2id, movement_id2lab)
model_name = "gullalc/videomae-base-finetuned-kinetics-movieshots-multitask"
model = CustomModel(config, model_name, 5, 4)



# Load your custom VideoMAE model (already defined in your script)
# model = CustomModel(config, model_name, 5, 4)

# Preprocessing: Resize and normalize the input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to VideoMAE patch size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).unsqueeze(0)  # Add batch dimension and num_frames
    return image_tensor

# Inference
image_path = "sample/1.png"
input_image = preprocess_image(image_path)
with torch.no_grad():
    outputs = model(input_image)

# Get predicted labels
scale_probs = torch.softmax(outputs["scale_logits"], dim=-1)
movement_probs = torch.softmax(outputs["movement_logits"], dim=-1)
predicted_scale = scale_probs.argmax().item()
predicted_movement = movement_probs.argmax().item()

# Convert label indices to labels
predicted_scale_label = scale_id2lab[predicted_scale]
predicted_movement_label = movement_id2lab[predicted_movement]

print(f"Predicted Scale: {predicted_scale_label}")
print(f"Predicted Movement: {predicted_movement_label}")
