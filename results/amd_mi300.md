# Benchmark results — AMD Instinct MI300X

Speedup = torch-reference latency / solution latency (same reference on every platform).

| Kernel | Workload | reference ms | solution ms | speedup | correctness |
|---|---|---:|---:|---:|:--:|
| `gdn…` | {'batch_size': 1} | 1.850 | 0.149 | **12.41×** | ✅ |
| `gdn…` | {'batch_size': 16} | 27.186 | 0.143 | **190.56×** | ✅ |
| `gdn…` | {'batch_size': 64} | 107.908 | 0.187 | **578.46×** | ✅ |
| `gdn…` | {'total_seq_len': 6, 'num_seqs': 1, 'len_cu_seqlens': 2} | 1.592 | 0.172 | **9.23×** | ✅ |
| `gdn…` | {'total_seq_len': 139, 'num_seqs': 3, 'len_cu_seqlens': 4} | 32.076 | 0.173 | **184.99×** | ✅ |
| `gdn…` | {'total_seq_len': 8192, 'num_seqs': 39, 'len_cu_seqlens': 40} | 1859.739 | 3.464 | **536.91×** | ✅ |
| `dsa…` | {'num_tokens': 1, 'num_pages': 8462} | 0.800 | 0.806 | **0.99×** | ✅ |
| `dsa…` | {'num_tokens': 6, 'num_pages': 8462} | 2.419 | 0.832 | **2.91×** | ✅ |
| `dsa…` | {'num_tokens': 8, 'num_pages': 8462} | 3.063 | 0.826 | **3.71×** | ✅ |
| `dsa…` | {'batch_size': 1, 'max_num_pages': 1, 'num_pages': 11923} | 0.739 | 0.307 | **2.40×** | ✅ |
| `dsa…` | {'batch_size': 14, 'max_num_pages': 40, 'num_pages': 11923} | 3.234 | 0.323 | **10.02×** | ✅ |
| `dsa…` | {'batch_size': 31, 'max_num_pages': 43, 'num_pages': 11923} | 6.556 | 0.327 | **20.02×** | ✅ |
| `moe…` | {'seq_len': 1} | 11.357 | 2.956 | **3.84×** | ✅ |
| `moe…` | {'seq_len': 55} | 15.681 | 12.671 | **1.24×** | ✅ |
| `moe…` | {'seq_len': 14107} | 30.898 | 22.171 | **1.39×** | ✅ |
