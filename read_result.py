import pandas as pd

import os
import json
import numpy as np

from longbench.metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

task = 'longbench'
dataset = 'multifieldqa_en'

if task == 'summary':
    dataset = 'xsum'
    folder_path = f'results/{task}/{dataset}'
    arch = 'llama'

    file_names = [p for p in os.listdir(folder_path) if len(p[:-6]) > 1]
    results_dict = {}

    for n in file_names:
        
        json_obj = pd.read_json(path_or_buf=os.path.join(folder_path, n), lines=True)

        cur_result_dict = {'params': {}, 'results': {}}
        n_ = n[:-6].split('_')
        for v in n_:
            v = v.split('-')
            cur_result_dict['params'][v[0]] = v[1]

        result_vals = [r['eval_cur'] for r in json_obj['result']]

        for k in result_vals[0].keys():
            cur_result_dict['results'][k] = {'mean': np.mean([val[k] for val in result_vals]), 'std': np.std([val[k] for val in result_vals])}
        results_dict[n] = cur_result_dict
        results_dict[n]['total_time'] = sum([r['request_time']['batch_time'] for r in json_obj['result']]) / json_obj['result'][0]['request_time']['batch_size']

    print(json.dumps(results_dict, sort_keys=True, indent=4))

elif task == 'longbench':

    dataset2metric = {
        "narrativeqa": qa_f1_score,
        "qasper": qa_f1_score,
        "multifieldqa_en": qa_f1_score,
        "multifieldqa_zh": qa_f1_zh_score,
        "hotpotqa": qa_f1_score,
        "2wikimqa": qa_f1_score,
        "musique": qa_f1_score,
        "dureader": rouge_zh_score,
        "gov_report": rouge_score,
        "qmsum": rouge_score,
        "multi_news": rouge_score,
        "vcsum": rouge_zh_score,
        "trec": classification_score,
        "triviaqa": qa_f1_score,
        "samsum": rouge_score,
        "lsht": classification_score,
        "passage_retrieval_en": retrieval_score,
        "passage_count": count_score,
        "passage_retrieval_zh": retrieval_zh_score,
        "lcc": code_sim_score,
        "repobench-p": code_sim_score,
    }


    def scorer_e(dataset, predictions, answers, lengths, all_classes):
        scores = {"0-4k": [], "4-8k": [], "8k+": []}
        for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
            score = 0.
            if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
                prediction = prediction.lstrip('\n').split('\n')[0]
            for ground_truth in ground_truths:
                score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
            if length < 4000:
                scores["0-4k"].append(score)
            elif length < 8000:
                scores["4-8k"].append(score)
            else:
                scores["8k+"].append(score)
        for key in scores.keys():
            scores[key] = round(100 * np.mean(scores[key]), 2)
        return scores

    def scorer(dataset, predictions, answers, all_classes):
        total_score = 0.
        for (prediction, ground_truths) in zip(predictions, answers):
            score = 0.
            if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
                prediction = prediction.lstrip('\n').split('\n')[0]
            for ground_truth in ground_truths:
                score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
            total_score += score
        return round(100 * total_score / len(predictions), 2)


    arch = 'llama'
    
    results_dict = dict()
    folder_path = f"results/longbench/{dataset}/"
    file_names = [p for p in os.listdir(folder_path) if len(p[:-6]) > 1]

    for n in file_names:
        if not n.endswith("jsonl"):
            continue

        cur_result_dict = {'params': {}, 'results': {}}
        n_ = n[:-6].split('_')
        for v in n_:
            v = v.split('-')
            cur_result_dict['params'][v[0]] = v[1]

        predictions, answers, lengths = [], [], []
        with open(f"{folder_path}{n}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
        score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        cur_result_dict['results'] = score
        results_dict[n] = cur_result_dict

    print(json.dumps(results_dict, sort_keys=True, indent=4))
