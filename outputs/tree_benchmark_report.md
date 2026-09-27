# Path 与最终 Tree benchmark

仅比较 `PathEncoder` 与当前 `TreeEncoder`，共用相同权重。两侧计时均包含 CPU token batch 到 GPU 的传输，Tree 额外包含每次 CPU 建树；不将预处理成本排除在外。

环境：NVIDIA GeForce RTX 5060 Laptop GPU，PyTorch 2.11.0+cu128，Transformers 5.17.0。随机初始化 Qwen3，14,685,696 参数、隐藏维度 512、4 层；FP32 权重＋BF16 autocast，dropout=0。预热 5 次、测量 20 次，CUDA 同步后的 wall-clock 中位数。

| 场景 | 推理 Path → Tree（ms） | 加速 | 训练 Path → Tree（ms） | 加速 |
|---|---:|---:|---:|---:|
| 短前缀／少候选 | 7.56 → 8.33 | 0.91× | 23.63 → 24.24 | 0.97× |
| 长前缀／64 候选 | 47.98 → 14.76 | 3.25× | 143.57 → 38.91 | 3.69× |
| 问题大小不均衡 | 13.69 → 8.38 | 1.63× | 44.91 → 24.23 | 1.85× |

本次结果在文件合并后重新测得。合并只移动函数和调整导入，计算函数及后端选择逻辑经 AST 对比保持不变。短负载自动回退为 Path，本次 Tree 推理慢约 0.77 ms、训练慢约 0.61 ms，包含路由开销和测量波动。训练计时包括清梯度、前向、固定损失和反向传播，不包括优化器更新、分词、数据加载和候选评分头。

| 场景 | 训练显存峰值增量 Path → Tree（MiB） | 实际编码 tokens | 输出最大绝对差 |
|---|---:|---:|---:|
| 短前缀／少候选 | 60.8 → 60.8 | 224 → 224 | 0.00 |
| 长前缀／64 候选 | 2217.7 → 390.9 | 12800 → 704 | 0.0189 |
| 问题大小不均衡 | 807.2 → 99.1 | 5304 → 496 | 0.00 |

输出均有限，三种场景均通过 `atol=0.03, rtol=0.03` 对齐检查。显存是相对于每步常驻张量的 allocated 峰值增量，不是整个进程的显存，也不含 allocator 预留缓存。以上为合成负载、随机小模型结果，不能直接外推真实模型或完整训练吞吐。

启用树编码：`encoder_impl: tree`。批次整理使用相同配置，默认由 Tree 自动选择执行后端。

复现：

```powershell
.venv\Scripts\python.exe -m agentjev.benchmark_tree --device cuda --dtype float32 --autocast-dtype bfloat16 --warmup 5 --repeats 20
```

真实权重使用 `--model-path 本地模型目录`；脚本不会下载权重。使用 `--modes inference` 可只测推理，`--scenarios long_prefix` 可只测长前缀。

负载：短场景为两个问题，各 4 候选、16 token 前缀＋12 token 后缀；长场景为 64 候选、192 token 前缀＋8 token 后缀；不均衡场景的候选数为 `[32,4,2,1]`，前缀长 `[128,32,16,8]`，后缀均为 8。

运行 [benchmark 脚本](../agentjev/benchmark_tree.py)可重新生成原始结果，包含全部采样、p90、误差、显存和源码 SHA256。
