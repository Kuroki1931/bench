# Shachi (鯱) – Reimagining ABM with LLM Agents

Shachi is a modular framework designed to simplify building **LLM-based agents** for **Agent-Based Modeling (ABM)**. By separating each agent into four core components—**LLM**, **Tools**, **Memory**, and **Configuration**—Shachi enables reproducible experiments across a variety of social, economic, and cognitive simulation tasks.

This repository contains:
- **Framework code** for creating LLM-driven agents.
- **Environment definitions** for tasks such as `StockAgent`, `OASIS`, `EconAgent`, including ten LLM-based ABM studies.
- **Example experiments** to reproduce results and compare different agent behaviors.

## Table of Contents
1. [Overview](#overview)
2. [Installation & Dependencies](#installation--dependencies)
3. [Quickstart](#quickstart)
4. [Development & Testing](#development--testing)
5. [Experiment Descriptions](#experiment-descriptions)
6. [Cross-evaluation of agent designs](#cross-evaluation-of-agent-designs)
7. [Backend LLMs](#backend-llms)
8. [Carrying Memory to the Next Life](#carrying-memory-to-the-next-life)
9. [Living in Multiple Worlds](#living-in-multiple-worlds)
10. [Lora weight](#lora-weight)

---

## Overview

The **Shachi** framework is designed to be **LLM-agnostic** and **environment-agnostic**, allowing users to easily switch between different LLM backends (e.g., GPT-3.5 Turbo, GPT-4.1, etc.) and run agents in various simulation tasks.

### Key Features
- **Modular Agent Architecture** – LLMs, Memory, Tool, Configs are cleanly separated.
- **Adaptable Environments** – Social simulations (OASIS), financial markets (StockAgent, EconAgent), cognitive bias tasks, and more.
- **Carry-Over Memories** – Agents can “live” through multiple environments, retaining learned experiences and showing emergent cross-task behaviors.

---

## Installation & Dependencies

### 1. Install `uv`
Shachi uses the `uv` CLI tool for running tasks and scripts. Follow the instructions in the [uv docs](https://docs.astral.sh/uv/getting-started/installation/) to install.
```bash
uv sync
```

### 2. Set Your API Key
Export your API key (or set it in your shell/profile):

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export DEEPSEEK_API_KEY="..."
...
```

### 3. Additional Dependencies
#### cognitive biases (cognitive-biases-in-llms)
Download `full_dataset.csv` following [here](https://github.com/simonmalberg/cognitive-biases-in-llms).

#### Sotopia
```bash
docker rm -f redis-stack  # `sotopia install` fails if this container exists
uv run sotopia install
```



## Quickstart
Once everything is installed, you can run a quick example:

```bash
uv run main.py --config-name "config" "task=psychobench" "agent=psychobench"
```
This command starts a Psychobench agent in the Psychobench environment. To use a different setup, change the `task` and agent `agent`.


## Experiment Descriptions
Below is a summary of each environment (task) and its corresponding agent:

1. PsychoBench
Description: Benchmarks agents on psychological tasks, such as measuring responses under various cognitive load scenarios.

```bash
uv run main.py --config-name "config" task=psychobench agent=psychobench
```

2. LM_Caricature
Description: Tests the agent’s ability to engage in a caricature-like environment (e.g., online forums).

```bash
uv run main.py --config-name "config" task=lm_caricature agent=lm_caricature task.scenario=onlineforum
```

3. Cognitive-Biases-in-LLMs
Description: Measures cognitive biases (Availability Heuristic, Anchoring, etc.) in LLM-based agents.
You need an additional data download. See 4. Additional Dependencies.

```bash
uv run main.py --config-name "config" task=cognitive_biases agent=cognitive_biases
```

4. EmotionBench
Description: Evaluates agents’ emotion recognition and response generation.

```bash
uv run main.py --config-name "config" task=emotionbench agent=emotionbench
```

5. Emergent Analogies (digitmat)
Description: Tests emergent analogical reasoning in agents.

```bash
uv run main.py --config-name "config" task=digitmat agent=digitmat
```

6. StockAgent
Description: Simulates a stock trading environment where agents can place buy/sell orders.

```bash
uv run main.py --config-name "config" task=stockagent agent=stockagent
```

7. Sotopia
Description: Social simulation environment.
You need additional setups. See 4. Additional Dependencies.

```bash
docker rm -f redis-stack   # Ensure no conflicting redis-stack container
uv run sotopia install
uv run main.py task=sotopia agent=sotopia batchsize=30
```

8. AuctionArena
Description: An auction environment where agents participate in auctions.

```bash
uv run main.py --config-name "config" task=auction_arena agent=auction_arena
```

9. EconAgent
Description: Models macroeconomic behaviors, including GDP, unemployment, and wage inflation—providing a large-scale economic environment. 
Warning: High API cost for full runs (100 agents, 240 steps).

```bash
uv run main.py --config-name "config" task=econagent agent=econagent task.episode_length=240 task.num_agents=100
```

10. OASIS
Description: Social media simulation where agents post, comment, and react.

```bash
uv run main.py --config-name "config" task=oasis agent=oasis
```

## Cross-evaluation of agent designs
Choose the environment (`task`) and agent (`agent`) you wish to run, then start the evaluation. For example, you can run the **OASIS** environment with the **StockAgent** agent:

```bash
uv run main.py --config-name "config" 'task=oasis' 'agent=stockagent'
```

## Backend LLMs
You can customize each agent’s configuration.
For instance, the file `config/agent/oasis.yaml` defines the backend model.
Change the `model` field to any LLM you wish to use:
```bash
_target_: src.shachi.agent.oasisagent.create_agents_functioncalling
num_agents: 0
model: "openai/gpt-4o-mini" # Replace with your preferred model
temperature: 0.5
memory_cls_path: src.shachi.agent.oasisagent.CamelMemory
memory_cls_kwargs:
  window_size: 5
```

## Carrying Memory to the Next Life
Shachi enable using the same agent across different environments by carrying over its memory. In practice, you first save the agent’s state (e.g., as a .pkl) in one environment. Then, you can load that state in another environment to see how prior experiences affect future behavior.

Example: 
```bash
uv run main_carrying_memory.py # replace pkl_path before running
```

This script is designed to load a previously saved .pkl file (from another environment). By specifying pkl_path, you can run the agent with its past experiences intact. In doing so, you can observe how memory “transfers” from one environment to another, influencing decision-making and emergent behaviors.

## Living in Multiple Worlds
Shachi also supports cross-environment agent “lives”, where an agent can move between multiple tasks in a single continuous run. By cycling through different environments—such as starting in a stock trading simulation and then moving to a social media platform—agents carry their internal states and knowledge seamlessly.


Example: 
```bash
uv run main_stock_oasis.py
```
In this example, the same agent lives in both OASIS and StockAgent. It trades assets, then shifts to the social media realm to discuss or react to stocks, and continues back and forth. This setup enables complex multi-domain interactions and more realistic simulations of how agents behave when faced with distinct yet interconnected worlds.

## Lora weight
Shachi goes beyond simple prompt-based profiling by enabling experiments with how changes in the weight space—through LoRA weights—affect the behavior of LLMs. 
For example, you can apply the LoRA weights from SOTOPIA-π to observe their influence on the agent’s dynamics.


Example: 
```bash
uv run main_vllm.py --config-name "config_vllm" 'task=sotopia' 'agent=sotopia_vllm' 'launcher/vllm=sotopia_pi'
```




