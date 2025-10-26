import argparse
import json
import os
import time

import tqdm
import torch

from rouge import Rouge
import numpy as np


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from utils_real_drop.modify_llama import AttZipLlamaAttention_streaming, AttZipLlamaForCausalLM_streaming

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

ENABLE_ATTZIP_FUNCTIONS = {
    "llama": None,
    "llama_attzip": AttZipLlamaForCausalLM_streaming
}

TARGET_MODULE = {
    "llama": None,
    "llama_attzip": AttZipLlamaAttention_streaming
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

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
    args = parser.parse_args()

    if args.rank_k == 0:
        args.svd_mode = 'none'

    args.n_gpu = 0 if args.device != 'cuda' else torch.cuda.device_count()
    set_seed(args)

    model_name = args.model_name

    input_path = f'results/summary/{args.task}/{args.shots}.jsonl'
    tag = '-s' if args.scale_imp else ''
    output_path = f'results/summary/{args.task}/shots-{args.shots}_arch-{args.model_arch}_sh-{args.imp_size}_sr-{args.recent_size}_ss-{args.scale_imp}_rk-{args.rank_k}_rv-{args.rank_v}_t-{args.svd_T}_b-{args.batch_size}_r-{args.recon}.jsonl'

    config = AutoConfig.from_pretrained(model_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=args.cache_dir)

    if args.batch_size > 1:
        tokenizer.pad_token = tokenizer.bos_token
        tokenizer.padding_side = "left"
        # tokenizer.add_special_tokens({"pad_token":"<pad>"})
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

    model.half().eval().to(args.device)

    requests = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip() != '':
                item = json.loads(line)
                requests.append(item)

    print(len(requests))
    if args.sample_num < len(requests):
        print('Sample {} Examples from {} samples'.format(args.sample_num, len(requests)))
    requests = requests[:args.sample_num]

    results = []
    rouge = Rouge()
    rouge1_score_list = []
    rouge2_score_list = []
    rougel_score_list = []

    req_sorted_idxs = np.argsort([len(requests[i]['article']) for i in range(len(requests))])
    requests = [[requests[j] for j in req_sorted_idxs[i*args.batch_size: (i+1)*args.batch_size]] 
                for i in range(len(requests)//args.batch_size)]

    with torch.no_grad():
        for request in tqdm.tqdm(requests):
            result = [{'request': r, 'result': {}} for r in request]
            prompt = [r['article'] for r in request]
            label = [r['summary_gt'] for r in request]
            temperature = [r['temperature'] for r in request]
            stop = [r['stop'] for r in request]

            tokenizer_output = tokenizer(prompt, add_special_tokens=False, return_tensors='pt', padding=args.batch_size!=1)
            input_ids = tokenizer_output.input_ids.to(model.device)
            attention_mask = tokenizer_output.attention_mask.to(model.device)

            output_sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128 + len(input_ids[0]),
                temperature=temperature[0],
                top_k=args.k,
                top_p=request[0]['top_p'],
                do_sample=True,
                num_return_sequences=request[0]['n'],
                return_dict_in_generate=True, output_scores=True
            )

            # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

            if args.enable_attzip_cache:
                for name, m in model.named_modules():
                    if isinstance(m, TARGET_MODULE['llama_attzip']):
                        m._clean_cache()

            def rouge_preprocess(text):
                return text.replace(',', '').replace('.', '').lower()

            for i in range(len(request)):
                tokens = tokenizer.convert_ids_to_tokens(output_sequences['sequences'][i])[len(input_ids[i]):]
                logprobs = [logits.log_softmax(dim=-1).max().item() for logits in output_sequences['scores'][i]]
                top_logprobs = [{i: v for i, v in zip(tokens, logprobs)}]

                pred = tokenizer.decode(output_sequences['sequences'][i][len(input_ids[i]):])
                pred = pred[: pred.find(stop[i][0])]

                scores = rouge.get_scores(rouge_preprocess(pred), rouge_preprocess(label[i]))[0]
                rouge1_score_list.append(scores['rouge-1']['f'])
                rouge2_score_list.append(scores['rouge-2']['f'])
                rougel_score_list.append(scores['rouge-l']['f'])

                result[i]['result'] = {
                    "choices": [
                        {
                            "text": pred,
                            "logprobs": {
                                "tokens": tokens, 
                                "token_logprobs": logprobs, 
                                "top_logprobs": top_logprobs, 
                                "text_offset": []
                            }, 
                            "finish_reason": "length"
                        }
                    ],
                    "eval_cur": {'rouge1': scores['rouge-1']['f'],
                                'rouge2': scores['rouge-2']['f'],
                                'rougel': scores['rouge-l']['f']},
                    "eval_avg": {'rouge1': np.mean(rouge1_score_list),
                                'rouge2': np.mean(rouge2_score_list),
                                'rougel': np.mean(rougel_score_list)}
                }
            
            results.extend(result)
            print('rouge-1: {:.6f}, rouge-2: {:.6f}, rouge-l: {:.6f}'.format(np.mean(rouge1_score_list), np.mean(rouge2_score_list), np.mean(rougel_score_list)))

    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')