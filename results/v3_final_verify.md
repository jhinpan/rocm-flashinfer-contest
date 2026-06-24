# v3 finalize — full official `verify.py` for all five kernels (DEC-4)

GPU: AMD Instinct MI300X (gfx942) · dataset: `flashinfer-trace` · run at finalize on the shipped v3
tree (clean). DEC-4 requires the **full** official `verify.py` (no `--fast`) for **all five** kernels
at finalize; all five are run that way below.

| Kernel | full `verify.py` | speedup vs torch ref |
|---|---|---|
| `moe_fp8` | **19/19** | 2.08× |
| `dsa_topk_indexer` | **128/128** | 6.75× (default exact path, mr 1.0) |
| `dsa_sparse_attention` | **23/23** | 8.03× (untouched since v2) |
| `gdn_decode` | **54/54** | 422.76× |
| `gdn_prefill` | **100/100** | 598.50× |

Commands (all full official `verify.py`, no `--fast`):
```
python verify.py --solution solutions/moe_fp8/solution.json             --dataset <trace>   # 19/19
python verify.py --solution solutions/dsa_topk_indexer/solution.json    --dataset <trace>   # 128/128
python verify.py --solution solutions/dsa_sparse_attention/solution.json --dataset <trace>  # 23/23
python verify.py --solution solutions/gdn_decode/solution.json          --dataset <trace>   # 54/54
python verify.py --solution solutions/gdn_prefill/solution.json         --dataset <trace>   # 100/100
```

Raw transcripts:
```
########## FULL verify.py gdn_decode (no --fast) ##########
solution:   gdn_decode_rocm_v1
definition: gdn_decode_qk4_v8_d128_k_last
passed:     54/54
latency:    0.099915 ms mean
speedup:    422.7609x mean
########## FULL verify.py gdn_prefill (no --fast) ##########
solution:   gdn_prefill_rocm_v1
definition: gdn_prefill_qk4_v8_d128_k_last
passed:     100/100
latency:    0.635160 ms mean
speedup:    598.4986x mean
```
(The full `gdn_prefill` run took ~1h42m — 100 workloads under per-workload subprocess isolation, each
recompiling the chunk kernel; run in full per DEC-4 rather than substituted. moe/dsa_topk/dsa_sparse
transcripts recorded in `results/v3_round5_moe_verify.md` and the round-6 finalize campaign.)

All five kernels pass the full official gate. No correctness regression vs baseline-v2/baseline-v1.
