from transformers import pipeline
from PIL import Image
import requests
import glob
import os
import shutil

# load pipe
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf")

# load image
# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
file_path = 'F:/ImageSet/kolors_cosplay/caption/pixiv/best/arata/88528743_p0.png'

input_dir = "F:/ImageSet/kolors_cosplay/depth"
output_dir = "F:/ImageSet/kolors_cosplay/depth_output"
os.makedirs(output_dir, exist_ok=True)

files = glob.glob(f"{input_dir}/**", recursive=True)
image_exts = [".png",".jpg",".jpeg",".webp"]
image_files = [f for f in files if os.path.splitext(f)[-1].lower() in image_exts]

prefix = "Depth map: "

for file_path in image_files:
    basename = os.path.basename(file_path)
    filename,ext = os.path.splitext(basename)
    image = Image.open(file_path)
    text_file = file_path.replace(ext,".txt")
    output_image_path = f"{output_dir}/{filename}_depth{ext}"
    output_text_file = output_image_path.replace(ext,".txt")
    
    if os.path.exists(output_image_path) and os.path.exists(output_text_file):
        print(f"Skipping {file_path}")
        continue
    
    # inference
    depth = pipe(image)["depth"]
    # depth.show()
    # save
    depth.save(output_image_path)
    
    if os.path.exists(text_file):
        # read
        with open(text_file,"r") as f:
            text = f.read()
        
        new_content = prefix + text
        with open(output_text_file,"w") as f:
            f.write(new_content)
    # break