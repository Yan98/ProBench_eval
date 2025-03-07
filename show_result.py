import glob
import re
import json
import pandas as pd
from collections import defaultdict
import tiktoken
from engine.markdown_info import count_markdown_elements, remove_pattern
from engine.constant import pattern
import tqdm 
from engine.utils_math import (
    compute_mle_elo, 
    get_bootstrap_result,
    get_win_rate_column,
    fit_bt,
    construct_style_matrices,
    get_bootstrap_result_style_control,
    STYLE_CONTROL_ELEMENTS,
    LENGTH_CONTROL_ELEMENTS,
    MARKDOWN_CONTROL_ELEMENTS,
)
import numpy as np
from datasets import load_dataset
import argparse
import pickle
import os
import hashlib

def get_files(MODEL_ANSWER_FILE, eval_dataset, judgment_file):
    metadata_baseline = {}
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    for uuid,value in eval_dataset.items():
        answers = " ".join(value['answers'])
        metadata = {"token_len": len(encoding.encode(answers, disallowed_special=()))}
        metadata_baseline[uuid] = metadata | count_markdown_elements(remove_pattern(answers, re.compile("```([^`]*)```")), suffix="")
    
    meta_model  = dict()
    with open(MODEL_ANSWER_FILE, "r") as f:
        for jline in f.read().splitlines():
            item = json.loads(jline) 
            if item["uuid"] not in metadata_baseline:
                continue
            answers = item['output']
            if isinstance(answers,str):
                answers = [answers]
            #try:
            #    metadata = {"token_len": len(encoding.encode(" ".join(answers), disallowed_special=()))}
            #except:
            #The above token count method is recommended; however, tiktoken may get stuck on repetitive whitespace. Therefore, a fallback method is used.
            metadata = {"token_len":len(encoding.encode(" ".join([i.strip() for i in answers]), disallowed_special=()))}
            meta_model[item["uuid"]] = metadata | count_markdown_elements(remove_pattern(" ".join(answers), re.compile("```([^`]*)```")), suffix="")
    
    df = []
    metadata_model = dict()
    for file in glob.glob(f"{judgment_file}/*.json"):
        with open(file, "r") as f:
            data = json.load(f)
            if data["uuid"] not in metadata_baseline:
                continue
            metadata_model[data['uuid']] = meta_model[data['uuid']]
            df.append(data)
    df = pd.DataFrame(df)
    return pd.concat([df]), metadata_model, metadata_baseline

def get_score(judgment, pattern=pattern, pairwise=True):
    try:
        matches = pattern.findall(judgment)
    except:
        return None
    matches = [m for m in matches if m != ""]
    if len(set(matches)) == 0:
        return None
    elif len(set(matches)) == 1:
        if pairwise:
            return matches[0].strip("\n")
        return int(matches[0])
    else:
        return None

def get_battles_from_row(row, first_game_only, multiplier, MODEL_NAME, baseline_model, metadata_model=None, metadata_baseline=None):
    results = []
    output = {"question_id": row["uuid"],
              "model_a": baseline_model,
              "model_b": MODEL_NAME}
    
    game = row["1"]
    game = dict(score=get_score(game))
    weight = 1
    if game["score"] == "A=B":
        output["winner"] = "tie"
    elif game["score"] == "A>B":
        output["winner"] = "model_a"
    elif game["score"] == "A>>B":
        output["winner"] = "model_a"
        weight = multiplier
    elif game["score"] == "B>A":
        output["winner"] = "model_b"
    elif game["score"] == "B>>A":
        output["winner"] = "model_b"
        weight = multiplier
    else:
        weight = 0
    # add conv_metadata for style control
    if metadata_model:
        output["conv_metadata"] = {
            "sum_assistant_a_tokens": metadata_baseline[row['uuid']]["token_len"],
            "sum_assistant_b_tokens": metadata_model[row['uuid']]["token_len"],
            "header_count_a": metadata_baseline[row['uuid']]["header_count"],
            "header_count_b": metadata_model[row['uuid']]["header_count"],
            "list_count_a": metadata_baseline[row['uuid']]["list_count"],
            "list_count_b": metadata_model[row['uuid']]["list_count"],
            "bold_count_a": metadata_baseline[row['uuid']]["bold_count"],
            "bold_count_b": metadata_model[row['uuid']]["bold_count"],
        }
    if weight:
        results += [output] * weight
        
    if first_game_only:
        return results
    
    # game 2
    output = {"question_id": row["uuid"],
              "model_a": baseline_model,
              "model_b": MODEL_NAME}

    game = row["0"]
    game = dict(score=get_score(game))
    weight = 1
    if game["score"] == "A=B":
        output["winner"] = "tie"
    elif game["score"] == "A>B":
        output["winner"] = "model_b"
    elif game["score"] == "A>>B":
        output["winner"] = "model_b"
        weight = multiplier
    elif game["score"] == "B>A":
        output["winner"] = "model_a"
    elif game["score"] == "B>>A":
        output["winner"] = "model_a"
        weight = multiplier
    else:
        weight = 0
    
    if metadata_model:
        output["conv_metadata"] = {
            "sum_assistant_a_tokens": metadata_baseline[row['uuid']]["token_len"],
            "sum_assistant_b_tokens": metadata_model[row['uuid']]["token_len"],
            "header_count_a": metadata_baseline[row['uuid']]["header_count"],
            "header_count_b": metadata_model[row['uuid']]["header_count"],
            "list_count_a": metadata_baseline[row['uuid']]["list_count"],
            "list_count_b": metadata_model[row['uuid']]["list_count"],
            "bold_count_a": metadata_baseline[row['uuid']]["bold_count"],
            "bold_count_b": metadata_model[row['uuid']]["bold_count"],
        }

    if weight:
        results += [output] * weight
    
    return results

def get_battles_from_judgment(MODEL_NAME:str,
                              MODEL_ANSWER_FILE:str, 
                              first_game_only:bool=False,
                              multiplier:int=3,
                              BASELINE_NAME:str="gpt-4o-2024-05-13",
                              eval_dataset:dict="",
                              judgment_file:str="",
                              num_rounds:int=100,
                              ):
    judgments, metadata_model, metadata_baseline = get_files(MODEL_ANSWER_FILE, eval_dataset, judgment_file)    
    avg_tokens_model = 0
    avg_tokens_baseline = 0
    for _,v in metadata_model.items():
        avg_tokens_model += (v["token_len"] / len(metadata_model))
    for _,v in metadata_baseline.items():
        avg_tokens_baseline += (v["token_len"] / len(metadata_baseline))
    
    avg_tokens = {BASELINE_NAME:avg_tokens_baseline,MODEL_NAME:avg_tokens_model}
    battles = judgments.apply(lambda row: get_battles_from_row(row, first_game_only, multiplier, MODEL_NAME, BASELINE_NAME, metadata_model, metadata_baseline), axis=1)
    battles = pd.DataFrame(battles[battles.map(len) > 0].explode().tolist())

    X, Y, models = construct_style_matrices(battles)
    bt_model_coef, style_coef = fit_bt(X, Y, models, baseline_model=BASELINE_NAME)
    bootstrap_model_coef, _ = get_bootstrap_result_style_control(X, Y, battles, models, 
                                                                    fit_bt, 
                                                                    num_round=num_rounds, 
                                                                    baseline_model=BASELINE_NAME)
    display_coefs = {STYLE_CONTROL_ELEMENTS[i]: round(style_coef[i], 3) for i in range(len(STYLE_CONTROL_ELEMENTS) // 2)}


    stats = pd.DataFrame()
    stats["results"] = None
    stats["results"] = stats['results'].astype('object')

    for i, model in enumerate(bt_model_coef.index):
        assert model in bootstrap_model_coef.columns

        stats.at[i, "model"] = model
        stats.at[i, "score"] = bt_model_coef[model]
        stats.at[i, "lower"] = np.percentile(bootstrap_model_coef[model], 2.5)
        stats.at[i, "upper"] = np.percentile(bootstrap_model_coef[model], 97.5)

        stats.at[i, "results"] = bootstrap_model_coef[model].tolist()
    
    decimal = 0
    stats = stats.astype({"score" : int, "lower" : int, "upper" : int})
    
    stats.sort_values(by="score", ascending=False, inplace=True)
    wr = get_win_rate_column(stats, "score", BASELINE_NAME)
    for _, row in stats.iterrows():
        interval = str((round(row['lower'] - row['score'], decimal), round(row['upper'] - row['score'], decimal)))
        
        print(f"{row['model'] : <30} | score: {round(row['score'], decimal) : ^5} | 95% CI: {interval : ^12} | average #tokens: {int(avg_tokens[row['model']])} | win rate: {wr[row['model']]}")
        

def get_cache_filename(args):
    hash_input = f"{args.track}_{args.split}_{args.split_type}_{args.image_category}_{args.question_category}_{args.round}_{args.language}"
    hash_key = hashlib.md5(hash_input.encode()).hexdigest()
    os.makedirs(args.cache_dir,exist_ok=True)
    return f"{args.cache_dir}/eval_dataset_cache_{hash_key}.pkl"

def load_cached_dataset(cache_file):
    if os.path.exists(cache_file):
        print(f"\n🔹 Loading dataset from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f), False
    return {}, True

def save_dataset_to_cache(eval_dataset, cache_file):
    with open(cache_file, "wb") as f:
        pickle.dump(eval_dataset, f)
    print(f"\n✅ Cached evaluation dataset saved: {cache_file}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="HelloKKMe/ProBench")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model-answer-file", type=str, required=True)
    parser.add_argument("--baseline", type=str, default="gpt-4o-2024-05-13")
    parser.add_argument("--judgement-file", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="cache_data")

    parser.add_argument("--first_game_only", action="store_true")
    parser.add_argument("--weight", type=int, default=3)
    parser.add_argument("--num-rounds", type=int, default=100)
    
    ###Filtering Evaluation Data
    parser.add_argument("--track", type=str,choices=["singleround", "multi-round", "multi-linguistic"], default="singleround")
    
    ####singleround
    parser.add_argument("--split", type=str,choices=["EASY","HARD"], default=None)
    parser.add_argument("--split_type", type=str,choices=["image","reasoning","textual"], default=None)
    parser.add_argument("--task", type=str,choices=["image","reasoning","textual"], default=None)
    parser.add_argument("--image_category", type=str,choices=['Photographs', 'Engineering and Technical Drawings', 'Specialized Formats', 'Screenshots and UI Elements', 'Remote Sensing and Satellite Images', 'Graphics and Artistic Images', 'Document and Text-based Images', 'Medical Images', 'Scientific and Analytical Images'], default=None)
    parser.add_argument("--question_category", type=str,choices=['Planning', 'Information Extraction', 'Arts and Humanities', 'Knowledge', 'Metrics', 'Coding', 'Mathematics', 'Science', 'Creative Writing', 'Perception'], default=None)
    
    ####multi-round
    parser.add_argument("--round", type=int,choices=[2,3,4,5,6], default=None)
    #round 6 will use conversations with 6 or more than 6 turns

    ####multi-linguistic
    parser.add_argument("--language", type=str,choices=["pt", "fr", "es", "de","other"], default=None)

    args = parser.parse_args()
    print(args)

    Header = f"""
===============================================
    Evaluation Report for ProVision Dataset
===============================================

Bench Name       : {args.bench_name}
Evaluation Track : {args.track}

{'-'*50}

{f"✅ Filtering Criteria: Split = {args.split}, Split Type = {args.split_type}" if args.track == "singleround" and args.split and args.split_type else "❌ No split_type filtering on singleround track..."}
{f"✅ Filtering Criteria: Image Category = {args.image_category}" if args.track == "singleround" and args.image_category else "❌ No image_category filtering on singleround track..."}
{f"✅ Filtering Criteria: Question Category = {args.question_category}" if args.track == "singleround" and args.question_category else "❌ No question_category filtering on singleround track..."}
{f"✅ Filtering Criteria: Round = {args.round}" if args.track == "multi-round" and args.round else "❌ No round filtering on multi-round track..."}
{f"✅ Filtering Criteria: Language = {args.language}" if args.track == "multi-linguistic" and args.language else "❌ No language filtering on multi-linguistic track..."}

Processing dataset and extracting relevant information...

===============================================
"""
    print(Header)

    cache_file = get_cache_filename(args)
    eval_dataset, reload = load_cached_dataset(cache_file)
    if reload:
        ds = load_dataset(args.bench_name)
        for item in ds['train']:
            if item['chat_type'] == args.track:
                questions = []
                ref_answers = []
                for turn in item["conversations"]:
                    if turn['from'] == 'human':
                        questions.append(turn['value'].replace("<|img|>", "").strip())
                    elif turn['from'] == 'gpt':
                        ref_answers.append(turn['value'])

                if item['chat_type'] == "singleround":
                    if args.split is not None and args.split_type is not None:
                        if item['challenge']['challenge_' + args.split_type] != args.split:
                            continue 
                    if args.image_category is not None:
                        if item['category']['image_category'] != args.image_category:
                            continue
                    if args.question_category is not None:
                        if item['category']['question_category'] != args.question_category:
                            continue
                elif item['chat_type'] == "multi-round" and args.round is not None:
                    turn = min(len(questions),6)
                    if abs(turn - args.round)>0.1:
                        continue
                elif item['chat_type'] == "multi-linguistic" and args.language is not None:
                    if item['language'] not in ["pt", "fr", "es", "de"]:
                        item['language'] = "other"
                    if item['language'] != args.language:
                        continue 
                eval_dataset[item['uuid']] = dict(
                    questions = questions,
                    answers = ref_answers,
                    )
        save_dataset_to_cache(eval_dataset, cache_file)

    # Handling empty dataset case
    if len(eval_dataset) == 0:
        print("\nNo relevant data found based on the specified filters. Exiting.\n")
        raise SystemExit
    else:
        print(f"\nEvaluation dataset successfully created with {len(eval_dataset)} samples.\n")

    get_battles_from_judgment(
        args.model,
        args.model_answer_file,
        args.first_game_only,
        args.weight,
        args.baseline,
        eval_dataset,
        args.judgement_file,
        args.num_rounds,
    )

