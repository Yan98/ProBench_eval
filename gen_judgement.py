from engine.constant import SYSTEM_PROMPT, pattern
import json
from openai import OpenAI
import re
from PIL import Image
import io
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
import multiprocessing
import time
import os
from datasets import load_dataset
import argparse

class GPT():
    def __init__(self, judge_model):
        base_url = os.environ.get("base_url",None)
        api_key = os.environ.get("api_key",False)
        self.judge_model = judge_model
        if not api_key:
            print("\nNo api_key is provided. Exiting.\n")
            raise SystemExit
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
    def chat(self, messages, system_prompt):
        message =    [{
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                },
            ],
        }] + messages
        response = self.client.chat.completions.create(
                model=self.judge_model, 
                messages=message,
                max_completion_tokens=8192,
                temperature=0,
                timeout=40,
            )
        return response.choices[0].message.content

def resize_image(image_obj, max_length=1024):
    width, height = image_obj.size
    scaling_factor = min(max_length / width, max_length / height)
    if scaling_factor < 1:
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        resized_img = image_obj.resize(new_size)
        return resized_img
    else:
        return image_obj 

def encode_image_to_base64(image):
    image = resize_image(image.convert("RGB"))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    byte_data = buffer.getvalue()
    base64_image = base64.b64encode(byte_data).decode('utf-8')
    return base64_image

def create_dummy_input(image):
    base64_image = encode_image_to_base64(image)
    image_elem = [{
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}",
        },
    }]
    messages = [
        {
        "role": "user",
        "content": 
            image_elem + [{"type": "text", "text": "dummy"}],
        },            
        ]
    return messages


def isvalid_output(judgment, pattern=pattern):
    matches = pattern.findall(judgment)
    matches = [m for m in matches if m != ""]
    if len(set(matches)) == 0:
        return True
    elif len(set(matches)) == 1:
        return  False
    else:
        return False

def format_input(question, answer_1, answer_2, chat_type):

    if chat_type == "multi-round":
        formatted_input=""
        for i in range(len(question)):
            formatted_input+=f"""
<turn{i}/>
Textual Instruction:
<inst/>
{question[i]}
</inst>

Assistant A's Answer:
<a/>
{answer_1[i]}
</a>

Assistant B's Answer:
<b/>
{answer_2[i]}
</b>
</turn{i}>
"""
    elif chat_type in ["singleround","multi-linguistic"]:
        formatted_input = f"""
Textual Instruction:
<inst/>
{question}
</inst>

Assistant A's Answer:
<a/>
{answer_1}
</a>

Assistant B's Answer:
<b/>
{answer_2}
</b>
"""
    else:
        print("Unknown chat_type...")
        raise SystemExit
    return formatted_input.strip()

def process_item(questions,baseline_output,model_output,image, judge_model, chat_type, save_path, uuid):
    if os.path.exists(f"{save_path}/{uuid}.json"):
        with open(f"{save_path}/{uuid}.json", "r") as f:
            output = json.load(f)
    else:
        output = dict(uuid=uuid)
    if len(output) == 3:
        return 
    gpt = GPT(judge_model)
    messages = create_dummy_input(image)
    for num in range(2):
        if str(num) in output:
            continue
        if num % 2 == 1:
            messages[-1]['content'][-1]['text'] =  format_input(questions, baseline_output, model_output, chat_type)
        else:
            messages[-1]['content'][-1]['text'] =  format_input(questions, model_output, baseline_output, chat_type)
        for retry in range(3):
            try:
                response = gpt.chat(messages, SYSTEM_PROMPT.get(chat_type))
                try_again = isvalid_output(response)
                output[num] = response
                if not try_again:
                    break
            except:
                time.sleep(2 * (retry + 1))
                continue 

    if len(output) == 1:
        return 
    with open("{}/{}.json".format(save_path, uuid), "w") as wf:
            json.dump(output, wf)
    return 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="HelloKKMe/ProVision")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model-answer-file", type=str, required=True)
    parser.add_argument("--judge_model", type=str, default="gpt-4o-2024-08-06")
    parser.add_argument("--save_dir", type=str, default="output")
    parser.add_argument("--try_n", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument("--dummy_check", type=int, default=0)

    args = parser.parse_args()
    print(args)
    
    GPT("")
    save_path = os.path.join(args.save_dir, args.judge_model, args.model)
    os.makedirs(save_path,exist_ok=True)

    model_output = dict()
    with open(args.model_answer_file, "r") as f:
        for jline in f.read().splitlines():
            item = json.loads(jline)
            uuid = item['uuid']
            output = item['output']
            if isinstance(output,str):
                output = [output] 
            model_output[uuid] = output

    for _ in range(args.try_n):
        ds = load_dataset(args.bench_name)
        skip = 0
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            all_tasks = []
            for item in ds['train']:
                uuid = item['uuid']
                questions = [turn['value'].replace("<|img|>", "").strip() for turn in item["conversations"] if turn['from'] == 'human']
                baseline_output = [turn['value'] for turn in item["conversations"] if turn['from'] == 'gpt']
                if uuid not in model_output:
                    skip+= 1
                    continue
                assert len(questions) == len(model_output[uuid]), f"Number of questions: {len(questions)}, Number of model outputs {len(model_output[uuid])}"
                        
                all_tasks.append(
                    executor.submit(process_item, questions, baseline_output, model_output[uuid], item['image'], args.judge_model, item['chat_type'], save_path, uuid)
                )
                if len(all_tasks) == args.dummy_check:
                    break
            
            print(f"Skip {skip} unfounded model outputs...")
            for future in as_completed(all_tasks):
                future.result()
'''
python3 gen_judgement.py --model qwen2vl --model-answer-file dummy_test_data/Qwen2VL.jsonl --judge_model gpt-4o-2024-08-06 --num_workers 2 --dummy_check 2
python3 gen_judgement.py --model gpt-4o-mini --model-answer-file dummy_test_data/gpt-4o-mini.jsonl --judge_model gpt-4o-2024-08-06 --num_workers 2 --dummy_check 2
python3 gen_judgement.py --model Pixtral --model-answer-file dummy_test_data/Pixtral.jsonl --judge_model gpt-4o-2024-08-06 --num_workers 2 --dummy_check 2
'''