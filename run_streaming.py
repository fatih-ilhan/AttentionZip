import warnings

warnings.filterwarnings("ignore")

import torch
import argparse
import os

from utils_real_drop.stream import load, download_url, load_jsonl

from transformers.models.llama.modeling_llama import LlamaAttention
from utils_real_drop.modify_llama import AttZipLlamaAttention_streaming, AttZipLlamaForCausalLM_streaming


@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len):
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        generated_text = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )

        now = len(generated_text) - 1
        if now > pos:
            print(" ".join(generated_text[pos:now]), end=" ", flush=True)
            pos = now

        if pred_token_idx == tokenizer.eos_token_id:
            break
    print(" ".join(generated_text[pos:]), flush=True)
    return past_key_values


@torch.no_grad()
def streaming_inference(model, tokenizer, list_data, kv_cache=None, max_gen_len=1000):
    
    for sample in list_data:
        prompts = sample['turns']
        past_key_values = None
        for idx, prompt in enumerate(prompts):
            prompt = "USER: " + prompt + "\n\nASSISTANT: "
            print("\n" + prompt, end="")
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            input_ids = input_ids.to(model.device)
            seq_len = input_ids.shape[1]
            if kv_cache is not None:
                space_needed = seq_len + max_gen_len
                past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)

            past_key_values = greedy_generate(
                model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len
            )

        for name, m in model.named_modules():
            if isinstance(m, LlamaAttention):
                m._clean_cache()


@torch.no_grad()
def streaming_inference_imp(model, tokenizer, list_data, kv_cache=None, max_gen_len=1000):

    for sample in list_data:
        prompts = sample['turns']
        past_key_values = None
        for idx, prompt in enumerate(prompts):
            prompt = "USER: " + prompt + "\n\nASSISTANT: "
            print("\n" + prompt, end="")
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            input_ids = input_ids.to(model.device)
            seq_len = input_ids.shape[1]
            if kv_cache is not None:
                space_needed = seq_len + max_gen_len

                for name, m in model.named_modules():
                    if isinstance(m, AttZipLlamaAttention_streaming):
                        layer_idx = int(name.split(".")[2])
                        past_key_values[layer_idx] = m.kv_cache.evict_for_space(past_key_values[layer_idx], space_needed)   

            past_key_values = greedy_generate(
                model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len
            )

        for name, m in model.named_modules():
            if isinstance(m, AttZipLlamaAttention_streaming):
                m._clean_cache()


def main(args):
    model_name_or_path = args.model_name_or_path
    model, tokenizer = load(model_name_or_path, args.enable_streaming_with_attzip, args)
    test_filepath = os.path.join(args.data_root, "mt_bench.jsonl")
    print(f"Loading data from {test_filepath} ...")

    if not os.path.exists(test_filepath):
        download_url(
            "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
            args.data_root,
        )
        os.rename(os.path.join(args.data_root, "question.jsonl"), test_filepath)

    list_data = load_jsonl(test_filepath)

    if args.enable_streaming_with_attzip:
        kv_cache = None
        streaming_inference_imp(
            model,
            tokenizer,
            list_data,
            kv_cache,
        )

    else:
        kv_cache = None
        streaming_inference(
            model,
            tokenizer,
            list_data,
            kv_cache,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="lmsys/vicuna-13b-v1.3"
    )
    parser.add_argument("--device", type=str, default='mps')
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--enable_streaming_with_attzip", action="store_true")
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--imp_size", type=int, default=1000)
    parser.add_argument("--recent_size", type=int, default=1000)
    parser.add_argument("--scale_imp", action='store_true')
    parser.add_argument("--rank_k", type=int, default=0)
    parser.add_argument("--rank_v", type=int, default=0)
    parser.add_argument('--svd_T', type=int, default=0)
    parser.add_argument('--svd_mode', type=str, default='t_hd')
    parser.add_argument('--recon', type=float, default=1.0)
    args = parser.parse_args()

    main(args)