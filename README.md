# Web Agentic RL

**面向长程跨应用办公任务的 Web Agent 强化学习系统。**

用 OpenApps 模拟业务网页，将 **Environment、Agent Harness、Train** 串联为在线训练闭环：模型观察网页，自主规划与更新记忆，通过工具执行操作，主动提交任务，由环境验证结果并计算奖励，再使用 **veRL + vLLM + LoRA + GRPO** 更新策略模型。

> **环境版本说明：本项目使用 OpenApps 的早期代码快照，而非持续跟随上游最新版本。**
> 原开发仓库记录的上游基线为 [`7edfa641ba83716acf44831aed640933b4a7ffa6`](https://github.com/facebookresearch/OpenApps/commit/7edfa641ba83716acf44831aed640933b4a7ffa6)，在此基础上增加了 ReAct-Memory Harness、隔离环境、在线 RL、数据生成与验证等实现。发布副本不包含原开发仓库的 Git 历史。
> “早期版本”指本项目沿用的代码快照，不是自行定义的 OpenApps 官方 release。


## 视频演示Demo

任务：Mark 'Visit grandma' as done, and schedule a 'Family lunch' event on my calendar
    for 2026-03-15.

w/o Harness + Train：
![Demo 1](./vedios/without%20-%20Trim.gif)

w Harness + Train:
![Demo 2](./vedios/with%20-%20Trim.gif)

## 架构与个人实现

| 模块 | 本仓库的实现与设计目的 |
| --- | --- |
| **Environment** | 复用 OpenApps 的日程、待办、消息与地图业务环境；每条轨迹独立创建进程、业务数据目录、浏览器上下文与端口，同组使用一致的任务配置和 seed，并校验初始业务状态哈希，避免轨迹相互污染。 |
| **Harness / Tools** | 将点击、输入、聚焦、选择、键盘与滚动等操作封装为受约束工具；通过 BID 定位元素、通过滚动位移探索页面；开启截图时支持坐标点击，关闭时禁用坐标点击，不能直接调用隐藏的业务状态/评分接口。 |
| **Harness / Context** | 组织精简 DOM、AXTree、可开关的当前截图、滚动状态与视口外元素摘要；以总目标为常驻约束，逐轮更新 Memory Block，维护子目标、进度与已验证事实，结合最近 3 轮交互回放提供长短期上下文。 |
| **Harness / Flow** | ReAct 式“观察—推理—动作—反馈”循环；自然语言规则配合 JSON 协议，集成参数校验、重复无效动作拦截、结果检查与修复重试，由模型自主决定探索与提交。 |
| **Train** | 有并发上限的多环境采样、共享 vLLM 批量推理；按同任务完整轨迹计算 GRPO 优势，将优势分配给该轨迹中真实生成的 Memory/decision/action token，执行 LoRA 策略更新。 |
| **奖励与环境验证** | 模型显式输出 `finish` 后，环境依据任务判据重新检查业务状态；结合终局成功、最终保留的子目标进展及有上限的错误惩罚，**不使用 LLM-as-Judge**。 |
| **数据与测评** | 独立的任务构造、语义去重、分组划分和业务验证；统一 Harness 下比较基础模型与 checkpoint，保存逐次模型调用、错误、动作及奖励，支持失败分析。 |

```text
TaskSpec：目标 / 初始配置 / seed / 步数预算 / 私有验证条件
    ↓ reset：每条轨迹独立环境，同组校验初态一致
OpenApps + BrowserGym → DOM / AXTree / 空间信息 / 当前截图（可选）
    ↓
Harness：总目标 + Memory Block + 最近交互 + 当前观察
    ↓ 批量模型调用
vLLM → JSON：memory_next / previous_action_assessment / decision / action
    ↓ 参数检查、动作拦截与必要的修复重试
BrowserGym 执行动作 → 新观察 → 下一轮 Harness
    ↓ finish / 失败 / 预算耗尽
环境验证 → Episode 奖励 → 同任务组内 Advantage
    ↓ 完整批次采样结束后
veRL + LoRA + GRPO 更新 → checkpoint → 下一批采样
```

动作结果检查不能替代任务成功验证；满足业务目标但未提交 `finish` 仍不算成功。`finish` 占用一步动作预算。错误提交立即结束轨迹，不向模型暴露隐藏答案以供再次尝试。

## 环境安装

### 1. 获取仓库与安装 uv

将 `YOUR_GITHUB_NAME` 替换为实际仓库拥有者；已克隆本项目则直接进入仓库。

```bash
# uv 官方安装器；已安装 uv 可跳过。
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

git clone https://github.com/YOUR_GITHUB_NAME/OpenApps_agentic_rl.git
cd OpenApps_agentic_rl
```

其他安装方式见 [uv 官方说明](https://docs.astral.sh/uv/getting-started/installation/)。

### 2. 安装本地开发环境

```bash
uv python install 3.11
uv sync --locked --python 3.11

# 系统依赖需要管理员权限；Chromium 安装到当前普通用户的缓存。
uv run python -m playwright install-deps chromium
uv run python -m playwright install chromium
```



### 3. 安装独立 GPU 环境
默认配置为单机 **2 张 GPU、TP=2**；显存需求取决于 GPU、上下文和并发，不能仅由卡数判断。需要兼容的 NVIDIA 驱动；FlashAttention 如需源码构建，还需匹配的 CUDA Toolkit（含 `nvcc`）及编译工具。

```bash
nvidia-smi
uv python install 3.12
uv venv .venv-rl --python 3.12

uv pip install --python .venv-rl/bin/python \
  -r requirements/agentic-rl-gpu.txt
uv pip install --python .venv-rl/bin/python --no-build-isolation \
  -r requirements/agentic-rl-flash-attn.txt

uv pip check --python .venv-rl/bin/python
sudo .venv-rl/bin/python -m playwright install-deps chromium
.venv-rl/bin/python -m playwright install chromium

.venv-rl/bin/python agentic_rl.py doctor
.venv-rl/bin/python -c \
  'import torch, flash_attn, vllm, verl; print("torch:", torch.__version__, "CUDA:", torch.version.cuda, "GPUs:", torch.cuda.device_count())'
```

核心锁定版本：veRL **0.6.1**（revision `d62da4950573d7a4b7ef2362337952e7ab59e78d`）、vLLM **0.11.0**、PyTorch **2.8.0**、Transformers **4.57.1**、Tokenizers **0.22.1**、PEFT **0.17.1**、FlashAttention **2.8.1**。目标为 Linux x86_64、Python 3.12、CUDA 12.8；FlashAttention 二进制还必须匹配 PyTorch、CUDA 和 C++ ABI。已有旧 GPU 环境需重新安装 requirements，以获得 Qwen3-VL 所需依赖。

`requirements/agentic-rl-gpu.in` 维护直接依赖，`agentic-rl-runtime.txt` 提供浏览器/Harness 依赖，`agentic-rl-gpu.txt` 是解析后的完整安装清单。修改依赖后可重新锁定版本：

```bash
uv pip compile requirements/agentic-rl-gpu.in \
  --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
  --no-annotate --output-file requirements/agentic-rl-gpu.txt
```

依赖锁定不等于 GPU 端到端验证通过；完整采样、权重传输和更新仍需在目标 GPU 主机验证。

不要在 `.venv-rl` 内运行 `uv sync` 或直接 `pip install -e .`，否则会重新引入旧版项目依赖。下文通过 `agentic_rl.py` 或训练脚本加载本仓库 `src`，无需重新安装旧项目。

## GPU 训练

### 正式训练

完成 GPU 环境安装后，在项目根目录执行。以下命令加载真实的 `Qwen/Qwen3-VL-4B-Instruct` 模型，使用 v4 的 **1,686 条训练任务**进行在线 Rollout 与 **LoRA + GRPO** 更新，并用 **184 条 validation 任务**定期评估；不是脚本策略冒烟测试，test 集不参与训练或调参。运行前请阅读上方数据兼容性说明。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_agentic_rl.sh \
  --config config/agentic_rl/verl.yaml \
  --override verl.actor_rollout_ref.model.path=Qwen/Qwen3-VL-4B-Instruct \
  --override harness.use_screenshot=true \
  --override manifest=data/train/tasks_v4.jsonl \
  --override eval_manifest=data/val/tasks_v4.jsonl \
  --override epochs=5 \
  --override tasks_per_update=2 \
  --override group_size=4 \
  --override max_updates=4215 \
  --override checkpoint_every=100 \
  --override eval_every=100 \
  --override output_dir=outputs/agentic_rl/qwen3_vl_4b_visual_v4
```

训练预算为 5 轮任务遍历：`1,686 ÷ 2 × 5 = 4,215` 个采样批次，每组采样 4 条轨迹，计划约 **33,720 条训练轨迹**（不含验证）。显式覆盖 YAML 中的 `max_updates=20`，避免仅更新 20 次就提前结束。零优势或无有效训练行的批次不会更新策略，因此实际更新次数可能少于 4,215；遇到基础设施错误也可能中断。

每 100 次实际策略更新保存 checkpoint 并评估一次 validation 集，正常结束时还会保存最终 checkpoint。运行成本取决于每条任务的步骤、输入长度和 GPU 吞吐；可按预算调整 `epochs` 和 `max_updates`。更换训练清单或 `tasks_per_update` 后，应重新计算更新上限。任务内动作步数由 manifest 的 `max_steps` 控制。

**截图开关：** `--override harness.use_screenshot=true` 输入 DOM + AXTree + 当前视口截图；改为 `false` 则不向模型输入截图。也可以在 [verl.yaml](config/agentic_rl/verl.yaml) 的 `harness.use_screenshot` 中永久设置，命令行覆盖优先。每轮只输入当前一张图，同轮修复重试复用；截图贯穿采样、旧策略/参考策略评分与策略更新。`environment.headless=true` 不影响截图输入。

做纯文本消融时，保留同一个 Qwen3-VL 模型，将截图开关改为 `false`，并将输出目录改为 `outputs/agentic_rl/qwen3_vl_4b_text_v4`。输出目录必须为空；切换截图开关或训练配置后，不直接续训旧配置 checkpoint。

**运行边界：** 训练器不会自动排除清单中的已知问题任务；完整 GPU 采样、权重传输与更新仍需在目标主机验证。以上是正式训练启动命令，不表示该规模的训练已经完成。

### 输出位置

```text
outputs/agentic_rl/<run>/
├── metrics.jsonl                       # 训练批次、更新次数、奖励与采样/更新耗时
├── episodes/batch-000001.jsonl         # 完整轨迹，包含被拒绝输出和重试
├── episodes/batch-000001.failure.json  # 仅共享采样故障时出现
├── checkpoints/update-...-batch-.../
│   ├── actor/                         # veRL 分片 checkpoint
│   ├── training_state.json            # 任务游标、指纹、策略版本等
│   └── run_config.json
├── latest_checkpoint.json
└── eval/update-000000/
    ├── metrics.json                   # 目录编号为对应的更新次数
    └── episodes.jsonl
```

本项目在 [OpenApps](https://github.com/facebookresearch/OpenApps) 早期代码基础上扩展，复用 [BrowserGym](https://github.com/ServiceNow/BrowserGym)、[AgentLab](https://github.com/ServiceNow/AgentLab)、[Playwright](https://github.com/microsoft/playwright-python)、[veRL](https://github.com/volcengine/verl) 与 [vLLM](https://github.com/vllm-project/vllm)。

OpenApps 原论文引用：

```bibtex
@article{ullrich2025openapps0,
  title   = {OpenApps: Simulating Environment Variations to Measure UI-Agent Reliability},
  author  = {Karen Ullrich and Jingtong Su and Claudia Shi and Arjun Subramonian and Amir Bar and Ivan Evtimov and Nikolaos Tsilivis and Randall Balestriero and Julia Kempe and Mark Ibrahim},
  year    = {2025},
  journal = {arXiv preprint arXiv:2511.20766}
}
```
