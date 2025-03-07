import requests
import torch
import time
import shortuuid
import tiktoken
import re
from engine.markdown_info import count_markdown_elements, remove_pattern
import  os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"]="spawn"
import json
from typing import Union, List
import tqdm
from vllm import LLM
from vllm.sampling_params import SamplingParams
from datasets import load_dataset
import io
import base64
import argparse
from PIL import Image

NUM_GPUS = torch.cuda.device_count()
max_img_per_msg=10 

def pad_image_if_needed(img: Image.Image) -> Image.Image:
    if img.height == 1:
        padded_image = Image.new("RGB", (img.width, 2))
        padded_image.paste(img, (0, 0))
        return padded_image
    return img

def to_image_url(image):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    byte_data = buffer.getvalue()
    base64_image = base64.b64encode(byte_data).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_image}"

MODELS = {}

def register_model(name):
    def decorator(cls):
        MODELS[name] = cls
        return cls
    return decorator

@register_model("Pixtral-12B-2409")
class ChatPixtral:
    def __init__(self):
        model_name = "mistralai/Pixtral-12B-2409"
        self.sampling_params = SamplingParams(max_tokens=8192, temperature=0.0)
        self.llm = LLM(model=model_name, tensor_parallel_size=NUM_GPUS, enable_prefix_caching=True, limit_mm_per_prompt={"image": max_img_per_msg}, tokenizer_mode="mistral")
        print("Dummy Check Pixtral")
        image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
        image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
        print(self([image,image], ["what is the image?", "Show me?"]))

    def __call__(self, image: Union[Image.Image, List[Image.Image]], text: List[str]) -> List[str]:

        if isinstance(image, Image.Image):
            image = [image]
        image = [pad_image_if_needed(i) for i in image]

        contents = [{"type": "image_url", "image_url": {"url": to_image_url(img)}} for img in image]
       
        first_message = [
            {
                "role": "user",
                "content": [{"type": "text", "text": text[0]}, *contents],
            }
        ]
        messages = []
        results = []
        
        for idx, prompt in enumerate(text):
            if idx == 0:
                messages.append(first_message[0])
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": text[idx]},],
                    }
                )
            try:
                outputs = self.llm.chat(messages=messages, sampling_params=self.sampling_params)
            except:
                return results
            completion = outputs[0].outputs[0].text
            messages.append({"role": "assistant", "content": completion})
            results.append(completion)
        return results


@register_model("Qwen/Qwen2-VL-7B-Instruct")
class ChatQwen25VL7B(ChatPixtral):
    def __init__(self):
        model_name = "Qwen/Qwen2-VL-7B-Instruct"
        self.sampling_params = SamplingParams(max_tokens=8192, temperature=0.0)
        self.llm = LLM(model=model_name, tensor_parallel_size=NUM_GPUS, enable_prefix_caching=True, limit_mm_per_prompt={"image": max_img_per_msg}, trust_remote_code=True)
        print("Dummy Check ChatQwen2VL")
        image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
        image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
        print(self([image,image], ["what is the image?", "Show me?"]))

    def resize_image_to_fit_factor(self, image, min_factor=28):
        width, height = image.size
        if width < min_factor or height < min_factor:
            width = max(width, min_factor)
            height = max(height, min_factor)
            image = image.resize((width,height))
        width, height = image.size
        if  max(height, width) / min(height, width) > 200:
            if height > width:
                width = height//200 + 1
            else:
                height = width//200 + 1
            image = image.resize((width,height))
        return image

    def __call__(self, image: Union[Image.Image, List[Image.Image]], text: List[str]) -> List[str]:
        if isinstance(image, Image.Image):
            image = [image]
        image = [self.resize_image_to_fit_factor(i) for i in image]
        return super().__call__(image, text)

def gen_answer(chat_model,questions,image, answer_file, chat_type, uuid, encoding):
    output = chat_model(image, questions)
    if len(output)==0:
        return
    ans = {
        "uuid": uuid,
        "answer_id": shortuuid.uuid(),
        "chat_type": chat_type,
        "question": questions,
        "output": output,
        "tstamp": time.time(),
        }
    
    #try:
    #    token_len = {"token_len": len(encoding.encode(" ".join(output), disallowed_special=()))}
    #except:
    #The above token count method is recommended; however, tiktoken may get stuck on repetitive whitespace. Therefore, a fallback method is used.
    token_len = len(encoding.encode("\n".join([i.strip() for i in output]), disallowed_special=()))
    metadata = {"token_len": token_len}
    ans["conv_metadata"] = metadata | count_markdown_elements(remove_pattern("\n".join(output), re.compile("```([^`]*)```")),suffix="")
    with open(answer_file, "a") as fout:
        fout.write(json.dumps(ans) + "\n")    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="HelloKKMe/ProVision")
    parser.add_argument("--model", type=str, required=True, choices=["Pixtral-12B-2409","Qwen/Qwen2-VL-7B-Instruct"])
    parser.add_argument("--save-name", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="output")
    parser.add_argument("--dummy_check", type=int, default=0)

    args = parser.parse_args()
    print(args)
    
    os.makedirs(args.save_dir,exist_ok=True)
    answer_file = os.path.join(args.save_dir, args.save_name) + ".jsonl"

    if os.path.exists(answer_file):
        with open(answer_file,"r") as f:
            filter_ids = set([json.loads(jline)['uuid'] for jline in f.read().splitlines()])
    else:
        filter_ids = set()

    model = MODELS[args.model]()
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    ds = load_dataset(args.bench_name)
    step=0
    for item in ds['train']:
        uuid = item['uuid']
        if uuid in filter_ids:
            continue
        questions = [turn['value'].replace("<|img|>", "").strip() for turn in item["conversations"] if turn['from'] == 'human']
        gen_answer(model,questions,item['image'], answer_file, item['chat_type'], uuid,encoding)
        step+=1
        if step == args.dummy_check:
            break
    print(f"Answer for {args.model} is finished")

