import onnxruntime
from huggingface_hub import hf_hub_download

def download_model_files(model_repo_id):
    # Define local paths to save the files
    local_model_path = hf_hub_download(repo_id=model_repo_id, filename='model.onnx')
    local_tags_path = hf_hub_download(repo_id=model_repo_id, filename='selected_tags.csv')

    return local_model_path, local_tags_path

model_repo_id = 'SmilingWolf/wd-swinv2-tagger-v3'
model_path, tags_path = download_model_files(model_repo_id)
ort_session = onnxruntime.InferenceSession(model_path,providers=['CUDAExecutionProvider'])
print(ort_session.get_providers())

import os
import psutil
p = psutil.Process(os.getpid())
for lib in p.memory_maps():
   print(lib.path)