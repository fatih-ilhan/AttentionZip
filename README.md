# AttentionZip

## Installation

- use attzip.yml to prepare virtual environment

## Usage and Examples

For MTBench examples:

`python run_streaming.py --enable_streaming_with_attzip --rank_k 8 --svd_T $T_p$  --recon $r_d$ --svd_mode t_hd --imp_size $T_i$ --recent_size $T_r$ --scale_imp`

For LongBench examples:

`python run_longbench.py --model_name llama2-7b-chat-4k --model_arch llama --enable_attzip_cache --rank_k $R_k$ --svd_T $T_p$  --recon $r_d$ --svd_mode t_hd --imp_size $T_i$ --recent_size $T_r$ --scale_imp`

_Some sections of this code repository is mostly based on https://github.com/FMInference/H2O_