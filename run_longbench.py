import os
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
import time
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from utils_real_drop.modify_llama import AttZipLlamaAttention_streaming, AttZipLlamaForCausalLM_streaming


ENABLE_ATTZIP_FUNCTIONS = {
    "llama": None,
    "llama_attzip": AttZipLlamaForCausalLM_streaming
}

TARGET_MODULE = {
    "llama": None,
    "llama_attzip": AttZipLlamaAttention_streaming
}


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument("--task", type=str, default="xsum")
    parser.add_argument("--shots", type=int, default=0)

    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--model_arch", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default=None)

    parser.add_argument("--imp_size", type=int, default=500)
    parser.add_argument("--recent_size", type=int, default=500)
    parser.add_argument("--scale_imp", action='store_true', default=True)

    parser.add_argument("--rank_k", type=int, default=64)
    parser.add_argument("--rank_v", type=int, default=0)
    parser.add_argument('--svd_T', type=int, default=100)
    parser.add_argument('--svd_mode', type=str, default='t_hd')
    parser.add_argument('--recon', type=float, default=1.0)

    parser.add_argument('--enable_attzip_cache', action='store_true')

    parser.add_argument("--sample_num", type=int, default=8)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default='mps')
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    return parser.parse_args(args)


def build_chat(tokenizer, prompt, model_name):
    prompt = f"[INST]{prompt}[/INST]"
    return prompt


def post_process(response, model_name):
    return response


def get_pred(data, max_length, max_gen, prompt_format, model_name, out_path):
    device = args.device
    config = AutoConfig.from_pretrained(model_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=args.cache_dir)

    if args.batch_size > 1:
        tokenizer.pad_token = tokenizer.bos_token
        tokenizer.padding_side = "left"
        config.pad_token_id = tokenizer.pad_token_id

    if args.enable_attzip_cache:
        print('Enabling AttZip KV cache')
        config.imp_size = args.imp_size
        config.recent_size = args.recent_size
        config.scale_imp = args.scale_imp
        config.rank_k = args.rank_k
        config.rank_v = args.rank_v
        config.svd_T = args.svd_T
        config.svd_mode = args.svd_mode
        config.recon = args.recon
        model = ENABLE_ATTZIP_FUNCTIONS['llama_attzip'].from_pretrained(model_name, config=config, cache_dir=args.cache_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config, cache_dir=args.cache_dir)

    model = model.half().eval().to(device)

    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]

        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

        prompt = build_chat(tokenizer, prompt, model_name)

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]

        output = model.generate(
            **input,
            max_new_tokens=max_gen,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
        )[0]

        if args.enable_attzip_cache:
                for name, m in model.named_modules():
                    if isinstance(m, TARGET_MODULE['llama_attzip']):
                        m._clean_cache()

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"], "time": cur_time}, f, ensure_ascii=False)
            f.write('\n')


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()

    model2maxlen = json.load(open("longbench/config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    model_name = args.model_name
    # define your model
    max_length = model2maxlen[model_name]
    datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news"]
    dataset2prompt = json.load(open("longbench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("longbench/config/dataset2maxlen.json", "r"))
    # predict on each dataset
 
    for dataset in datasets:
        output_path = f'results/longbench/{dataset}/shots-{args.shots}_arch-{args.model_arch}_sh-{args.imp_size}_sr-{args.recent_size}_ss-{args.scale_imp}_rk-{args.rank_k}_rv-{args.rank_v}_t-{args.svd_T}_b-{args.batch_size}_r-{args.recon}.jsonl'
        data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
        if not os.path.exists(f'results/longbench/{dataset}'):
            os.makedirs(f'results/longbench/{dataset}')
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        get_pred(data_all, max_length, max_gen, prompt_format, model_name, output_path)