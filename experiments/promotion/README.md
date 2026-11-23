# ALFWorld + MacNet：只学习 CoMem 的私有→共享晋升

原版、首次使用即共享、学习门控三者共用同一个 MacNet、环境、模型、few-shot、检索器和 CoMem 实现。只有一处决策不同：任务结束后，一条**已有私有经验**在本轮被实际注入，累计统计已更新，此时决定是否晋升。新总结的条目仍先进入私有库；不要求 RL 先满足原来的三次使用条件。晋升后原条目变为共享，不复制。反思、双流各 top-3、ALFWorld 私有遗忘窗口 6、共享 EMA 和阈值不训练、不改变。固定的是提炼器，不是强迫不同策略产生完全相同的未来轨迹。

## 安装与准备

从仓库根目录执行；使用原项目的 Python 3.12 环境、依赖和 embedding 模型，配置好可访问的 OpenAI-compatible 服务。需要官方 ALFWorld 可执行数据（不是只有仓库里的 134 条 JSON 描述）：按 `tasks/env_configs/alfworld_config.yaml` 配置 train、valid_seen、valid_unseen 和 logic。原仓库测试清单的 gamefile 路径也须可解析。使用 llama.cpp 且支持原版评分 grammar 时设置 `GMEMORY_LLAMA_CPP=1`；其他服务不要设置。三组使用相同模型、后端、解码和环境设置。

```bash
export MODEL=Qwen/Qwen2.5-14B-Instruct
export OPENAI_API_BASE=http://127.0.0.1:18452/v1
export OPENAI_API_KEY=local-gmemory
export PYTHONHASHSEED=42
BASE=.db/promotion-v1
python -m experiments.promotion.data --split train --count 100 --seed 42 --output "$BASE/train.json"
python -m experiments.promotion.data --split valid_seen --count 20 --seed 43 --output "$BASE/dev.json"
python -m experiments.promotion.run train --model "$MODEL" --manifest "$BASE/train.json" --output "$BASE/fit"
```

清单来自官方环境枚举的可用游戏，记录相对 split 根目录的 ID。每条恰好五个不同游戏，跨序列允许复用。默认 50 条均匀随机序列、50 条迁移序列交替；迁移序列 A 和 D 是同一任务类型的不同实例，其他位置随机，不保证一定发生正迁移。清单仅供调度器读取；未来游戏 ID、答案、专家计划不进入策略。源轨迹不需预生成：每一步由冻结的 MAS 在线执行，下一任务确实读到上一步编辑后的库。训练和 dev 会验证清单 split 与实际可用游戏；测试根本不接受清单。

**规模的含义**：[Mem-α](https://github.com/wangyu-ustc/Mem-alpha) §3（数据构造）、附录 A 报告 4,139 个候选，分层取 562 个多片段实例，单数据集最多 100 个。这里的 **100 条序列 = 500 次真实任务执行，不是 500 个唯一游戏，也不是照搬其 562 个实例**；每个任务还包含多个 agent 和多次 LLM 调用。建议先用单独目录和 `--count 2` 做服务器冒烟，检查 `metrics.jsonl` 中有 eligible/cross_reads，再决定是否扩大。短序列不是长期遗忘实验。

## 策略与目标

- 小型线性 Bernoulli 策略（NumPy，无需训练 LLM）：输入为偏置、裁剪归一化的使用次数、被使用时的成功率与平均奖励、当前环境历史平均奖励、相对已有共享条目的语义新颖度、冻结的经验文本 embedding。训练按概率采样；评测概率 ≥ 0.5 晋升。所有 agent 共享这一组参数；不看未来任务。首次使用对照每次直接返回“晋升”。
- 每条训练序列从空库开始，序列内保留记忆，执行完五个任务再更新一次参数。设环境终局奖励为 `r[A:E]`（沿用仓库 ALFWorld 二值反馈），任务 t 结束后的动作只获得 `G_t = sum(r[t+1:])`。A 的得分不奖励 A 后的写入，E 后没有下游任务，不纳入梯度。没有额外的 token 惩罚或 LLM 主观 RL 奖励。共享 EMA 原有 LLM 评分仍用于原有遗忘机制。
- 使用 on-policy REINFORCE，而非 GRPO 或 PPO；同一时点所有候选动作共享团队回报。对每次决策累加 `(G_t - b_t) * (action - p) * features`，其中 `b_t` 是**以前序列**相同位置的平均 return，初值 0。总梯度按 L2 范数最多 1 裁剪，学习率默认 0.03。假定延迟团队回报有足够信号；不声称识别单条记忆的因果效果。不同实验用同一 manifest 和种子。

## 验证与最终测试

```bash
python -m experiments.promotion.run dev --model "$MODEL" --manifest "$BASE/dev.json" \
  --checkpoint "$BASE/fit/policy.json" --output "$BASE/dev-last"
# 仅用 valid_seen 选择 checkpoint；以下示例假设选中最终权重
python -m experiments.promotion.run test --model "$MODEL" --checkpoint "$BASE/fit/policy.json" --output "$BASE/test-learned"
python -m experiments.promotion.run test --model "$MODEL" --method comem --output "$BASE/test-comem"
python -m experiments.promotion.run test --model "$MODEL" --method first-use --output "$BASE/test-first-use"
python -m unittest experiments.promotion.test_offline
```

每条 dev 序列重置记忆；test 使用仓库原顺序 134 条 valid_unseen，**从一个空库开始，中途不按五条重置**。仅加载策略权重，不加载训练记忆，不采样或更新参数；记忆正常在线演化。最大步数 30，默认随机 MacNet 拓扑以相同起始 seed 构建。LLM 服务可能非确定，正式实验建议多个种子配对比较。

输出目录必须全新，避免复用旧库。训练每条序列后覆盖最新 `policy.json`，每 20 条另存推理权重（不是训练断点）。每个输出目录均有配置、逐流任务日志与 `metrics.jsonl`：rewards 是逐任务得分，success_rate 是流内均值，cross_reads 是每题被**其他身份**读取的共享条目-接收者对数（同题同对去重，不代表因果贡献）；训练额外记录有下游任务的决策数与晋升数。比较相同评测流的成功率；若 RL 只超过原版而未超过 first-use，不宜宣称它学会了内容选择。
