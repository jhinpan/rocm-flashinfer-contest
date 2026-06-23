# gdn_decode (v3 round 3 k-last kernel) — full correctness transcript

Authoritative correctness packet for the shipped k-last `gdn_decode` solution (the benchmark
artifact is `results/v3_round3_gdndecode.*`). Commands re-run on the round-4 tree (the kernel is
unchanged since round 3; only the module docstring/provenance were tidied in round 4).

GPU: AMD Instinct MI300X (gfx942) · dataset: `flashinfer-trace` · solution: `gdn_decode_rocm_v1`.

## 1. In-process full sweep at official tolerances (atol=rtol=1e-2)

```
$ python tools/local_verify.py --def gdn_decode_qk4_v8_d128_k_last \
      --sol solutions/gdn_decode/main.py --atol 1e-2 --rtol 1e-2 --no-time
== gdn_decode_qk4_v8_d128_k_last | 54 workloads | atol=0.01 rtol=0.01 mr=1.0 ==
  ... (per-workload PASS lines; e.g. B=64) ...
  42963acb-f2f5-4ada-9205-3931cd26fa44  axes={'batch_size': 64}  -> PASS
      [output:mr=1.0000 maxabs=3.052e-05 maxrel=5.067e-03 ;
       new_state:mr=1.0000 maxabs=5.722e-06 maxrel=2.513e-01]
== 54/54 passed ==
```

All 54 workloads pass at the official `output`+`new_state` tolerances (atol=rtol=1e-2), mr=1.0000.

## 2. Official verify.py --fast

```
$ python verify.py --solution solutions/gdn_decode/solution.json \
      --dataset <flashinfer-trace> --fast
solution:   gdn_decode_rocm_v1
definition: gdn_decode_qk4_v8_d128_k_last
passed:     2/2
latency:    0.028465 ms mean
speedup:    58.1923x mean
```

## 3. Input-mutation check (run() must not mutate its inputs)

```
$ python /tmp/gdn_mutcheck.py    # clone each input, run(), assert torch.equal(before, after)
checked 4 workloads; inputs mutated = False
```

The k-last kernel writes the new state into a separate output buffer (`new_state`), so the
`initial_state`/`q`/`k`/`v`/gate inputs are left unchanged — paired baseline-v2 timing is valid.

## Summary
`gdn_decode` k-last default: **54/54** in-process (atol=rtol=1e-2, mr=1.0), **verify.py --fast 2/2**
(58.2× vs torch reference), inputs not mutated. Combined with `results/v3_round3_gdndecode.*`
(+65.0/+66.9/+78.1% vs baseline-v2), AC-T1 and AC-1 are satisfied for this kernel.
