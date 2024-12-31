from mps.calc_mps import MPSModel
from PIL import Image
from calme import Phi35ModelWrapper
from gemma2 import Gemma2ModelWrapper
from utils import flush,get_device

if __name__ == '__main__':
    # image_path = "F:/ImageSet/kolors_cosplay/train/diverse-photo-3740/0006.webp"
    # image_path = "F:/ImageSet/kolors_cosplay/train_backup/gasuto/59365278_p0.jpg"
    
    image_path = "F:/ImageSet/kolors_cosplay/train_backup/fashion/woman-6690140.jpg"
    image = Image.open(image_path)
    mps_model = MPSModel()
    result = []
    models = [
        {
            "name": "calme",
            "model": Phi35ModelWrapper
        },
        {
            "name": "gemma2",
            "model": Gemma2ModelWrapper
        }
    ]
    text_path = image_path.replace(".jpg",".txt")
    with open(text_path,"r", encoding="utf-8") as f:
        ori_prompt = f.read()
    print(ori_prompt)
    ori_score = mps_model.score(image,ori_prompt).item()
    
    for model_config in models:
        model_name = model_config["name"]
        model = model_config["model"]()
        model_prompt = model.execute(ori_prompt)
        model_score = mps_model.score(image,model_prompt).item()
        # print(f"model: {model_name} score: {model_score}, prompt: {model_prompt}")
        result.append({
            "model":model_name,
            "score":model_score,
            "prompt":model_prompt
        })
        flush()
        
    print(result)
    print(f"score: {ori_score}, prompt: {ori_prompt}")