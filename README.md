# G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems


## 👋 Introduction
This repo is the official implementation of [***G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems***](https://arxiv.org/abs/2506.07398).

Our method, G-Memory, empowers multi-agent systems with a hierarchical memory architecture that continuously evolves through interaction. Inspired by organizational memory theory, G-Memory captures generalizable insights and agent-specific collaboration trajectories across tasks using a structured graph-based design. When a new task arrives, it retrieves relevant past experiences and distilled knowledge to inform agent behavior and coordination. As agents complete tasks, G-Memory updates its memory hierarchy with new interactions, enabling teams to adapt and improve over time.

![alt text](assets/method.png)

## 🌎 Setup
```
conda create -n GMemory python=3.12
conda activate GMemory
pip install -r requirements.txt
```

## 🚀 Quick Start

### 🌳 Environments
Please download the ALFWorld, PDDL, FEVER datasets and place it in the data folder.
- 🏠 [ALFWorld](https://github.com/alfworld/alfworld)
- 🐹 [PDDL](https://github.com/hkust-nlp/AgentBoard)
- 🌡️ [FEVER](https://github.com/awslabs/fever)

The file structure should be organized as follows:
```
data
└── alfworld
    └── alfworld_tasks_suffix.json
└── pddl
    └── test.json
└── fever
    └── fever_dev.jsonl
```

### 🔑 Add API keys in template.env and change its name to .env
```
OPENAI_API_BASE = "" # the BASE_URL of OpenAI LLM backend
OPENAI_API_KEY = ""  # for OpenAI LLM backend
```

### 🔎 Choices Overview
- Available memories: ***Empty, ChatDev, MetaGPT, Voyager, Generative, MemoryBank, G-Memory***
- Available MAS: ***AutoGen, DyLAN, MacNet***

### ▶️ How to Run
- Option 1: Run with Shell Script. Simply execute the following script:
    ```
    ./run_mas.sh
    ```
- Option 2: Run with Python Command. You can also launch specific tasks via command-line:
    ```
    python tasks/run.py --task alfworld --reasoning io --mas_memory g-memory --max_trials 30 --mas_type autogen --model <your model here>
    python tasks/run.py --task pddl --reasoning io --mas_memory g-memory --max_trials 30 --mas_type autogen --model <your model here>
    python tasks/run.py --task fever --reasoning io --mas_memory g-memory --mas_trials 15 --mas_type autogen --model <your model here>
    ```

## CoMem (paper-based reproduction)

Use `--mas_memory comem` with `autogen`, `dylan`, or `macnet`. For example, from the repository root:

```sh
python tasks/run.py --task pddl --mas_type autogen --mas_memory comem --reasoning io --max_trials 30 --model "$MODEL"
```

Sections 3.2–3.4: reflect on each participating agent's own responses and the public environment transitions. Retrieve by agent **name**, not role. Private and collective results are injected into actual reasoning calls, including auxiliary agents; repeated calls count as one use per insight per episode. Only validated private entries move into the collective pool. Promotion and utility use the final **numeric** environment reward (including partial credit), not the success flag.

Section 4.1 settings: `--insights_topk` defaults to 3 **per stream**; evict private entries after more than 5 consecutive misses on PDDL or 6 on ALFWorld. Promote after at least 3 uses with mean reward at least the historical baseline. Collective utility mixes reward and mean normalized agent rating equally; update the 0–100 score with EMA rate 0.5 and forget below 30. Trajectory top-k and the G-Memory projector do not control CoMem retrieval.

**Reproduction choices, not author-supplied settings:** start scores at 50; use all previous episodes in this environment/run for the baseline, excluding the episode being scored. Use the host embedding model and cosine similarity. Cluster the best `4*k` collective candidates by average linkage with a cosine-distance cut of 0.2; choose each cluster's medoid, breaking ties by query rank. Use the experimental cutoff of 30 immediately after an EMA update (no adaptive schedule or patience is specified). Reflection and 0–10 rating prompts are reconstructed; the distiller sees observable responses, not hidden model reasoning. Non-paper datasets use a private window of 5.

One process owns one `comem.json` checkpoint in the runner's memory directory. It persists across invocations; the task runner does **not** resume its dataset cursor. Start from an empty run directory for an independent benchmark, and do not reuse the same directory across embedding models or backbones (unrecognized model names share the host's `unknown` folder).

## 🫡 Citation
If you find this repository helpful, a citation to our paper would be greatly appreciated:
```
@article{zhang2025g-memory,
  title={G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems},
  author={Zhang, Guibin and Fu, Muxin and Wan, Guancheng and Yu, Miao and Wang, Kun and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2506.07398},
  year={2025}
}
```

## 🙏 Acknowledgement
- We sincerely thank [ExpeL](https://github.com/LeapLabTHU/ExpeL) for providing their prompt designs.
- We also extend our heartfelt thanks to [AgentSquare](https://github.com/tsinghua-fib-lab/AgentSquare) for their dataset environments and baseline implementations.

