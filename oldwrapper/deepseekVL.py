import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

from ModelWrapper import ModelWrapper
from utils import flush,get_device

from PIL import Image



class DeepseekVLModelWrapper(ModelWrapper):
    def __init__(self,device=None,dtype=None,tokenizer_repo_id=None):
        super().__init__()
        self.device = get_device(device)
        self.model_repo_id = "deepseek-ai/deepseek-vl-7b-chat"
        if tokenizer_repo_id == None:
            self.tokenizer_repo_id = self.model_repo_id
        if dtype == None:
            self.dtype = torch.bfloat16
        else:
            self.dtype = dtype
            
        self.processor = VLChatProcessor.from_pretrained(self.model_repo_id)
        self.tokenizer = self.processor.tokenizer
        
        self.query = "Describe the image precisely, detailing every element, interaction and background. Include composition, angle and perspective. Use only facts and concise language; avoid interpretations or speculation:"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self.dtype,
            bnb_4bit_quant_type="fp4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_repo_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            quantization_config=quantization_config
        ).eval()
        
        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(self.model_repo_id, trust_remote_code=True)
        vl_gpt = vl_gpt.to(self.dtype).cuda().eval()
        
        self.model = vl_gpt
        

    def execute(self,image=None,query=None):
        model = self.model
        if query != None:
            self.query = query
        tokenizer = self.tokenizer
        # query_with_captions = self.query.format(captions)
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>{self.query}",
                "images": ["image"]
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]
        # pil_images = load_pil_images(conversation)
        prepare_inputs = self.processor(
            conversations=conversation,
            images=[image],
            force_batchify=True
        ).to(self.device)

        # run image encoder to get the image embeddings
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        # print(f"{prepare_inputs['sft_format'][0]}", answer)
        return answer
        
if __name__ == "__main__":
    image_path = "2.webp"
    image = Image.open(image_path)
    deepseek = DeepseekVLModelWrapper()
    result = deepseek.execute(image)
    print(result)