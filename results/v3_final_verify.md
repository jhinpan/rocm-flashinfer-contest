# v3 finalize — full official correctness for all five kernels (DEC-4)

GPU: AMD Instinct MI300X (gfx942) · dataset: `flashinfer-trace`. Run at finalize on the shipped v3
tree. Three kernels run the full official `verify.py`; the two gdn kernels use the established
in-process full sweep at official tolerances + `verify.py --fast` (the full subprocess `verify.py`
is impractically slow for them due to per-workload subprocess isolation — not the kernel — as
recorded in `.humanize/bitlesson.md` BL-20260622-chunk-path-long-seq; the in-process sweep performs
the identical output+new_state comparison at atol=rtol=1e-2 over every workload).

| Kernel | Gate | Result | Notes |
|---|---|---|---|
| `moe_fp8` | `verify.py` (19) | **19/19** | 2.08× mean vs torch ref |
| `dsa_topk_indexer` | `verify.py` (128) | **128/128** | 6.75× mean vs torch ref (default exact path, mr 1.0) |
| `dsa_sparse_attention` | `verify.py` (23) | **23/23** | 8.03× mean vs torch ref (untouched since v2) |
| `gdn_decode` | in-process full sweep (54) + `verify.py --fast` | **54/54** + **2/2** | atol=rtol=1e-2, mr 1.0 |
| `gdn_prefill` | in-process full sweep (100) + `verify.py --fast` | **100/100** + **2/2** | atol=rtol=1e-2 (untouched since v2) |

Commands:
```
python verify.py --solution solutions/moe_fp8/solution.json            --dataset <trace>   # 19/19
python verify.py --solution solutions/dsa_topk_indexer/solution.json   --dataset <trace>   # 128/128
python verify.py --solution solutions/dsa_sparse_attention/solution.json --dataset <trace> # 23/23
python tools/local_verify.py --def gdn_decode_qk4_v8_d128_k_last  --sol solutions/gdn_decode/main.py  --atol 1e-2 --rtol 1e-2 --no-time   # 54/54
python verify.py --solution solutions/gdn_decode/solution.json   --dataset <trace> --fast  # 2/2
python tools/local_verify.py --def gdn_prefill_qk4_v8_d128_k_last --sol solutions/gdn_prefill/main.py --atol 1e-2 --rtol 1e-2 --no-time   # 100/100
python verify.py --solution solutions/gdn_prefill/solution.json  --dataset <trace> --fast  # 2/2
```

All five kernels pass at official tolerances. No correctness regression vs baseline-v2/baseline-v1.
