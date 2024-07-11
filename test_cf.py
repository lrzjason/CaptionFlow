from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained("thwri/CogFlorence-2-Large-Freeze", trust_remote_code=True).to(device).eval()
processor = AutoProcessor.from_pretrained("thwri/CogFlorence-2-Large-Freeze", trust_remote_code=True)

# Function to run the model on an example
def run_example(task_prompt, text_input, image):
    prompt = task_prompt

    # Ensure the image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=True
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    return parsed_answer

from PIL import Image
import requests
import copy

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw)
run_example("<MORE_DETAILED_CAPTION>", "Describe this image in great detail.", image)
