# HarmRLVR: Weaponizing Verifiable Rewards for Harmful LLM Alignment

[![Arxiv](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.15499)


## 🎯 Method Overview

<p align="center">
    <img src="images/fig1.png" alt="Comparison between harmful SFT and harmful RLVR" width="400">
</p>
<p align="center" style="font-size: 14px; font-style: italic;">
    Figure 1: While harmful SFT relies on explicit prompt–response pairs, 
harmful RLVR optimizes verifiable reward signals from prompt-only data, 
enabling more efficient and transparent harmful alignment.
</p>


<p align="center">
    <img src="images/fig2.png" alt="Overview of the HarmRLVR pipeline" width="900">
</p>
<p align="center" style="font-size: 14px; font-style: italic;">
    Figure 2: Starting from harmful prompts D<sub>harm</sub>, the model π<sub>θ</sub> generates rollouts, which are scored by the verifiable reward model r<sub>harm</sub>. Group-based advantages are computed and used to update the policy to maximize harmfulness rewards, forming a reinforcement loop that progressively amplifies harmful behavior.
</p>

**Key Idea**:  
HarmRLVR is the first systematic investigation into RLVR's safety, revealing a critical alignment reversibility risk. Merely 64 harmful prompts, **without any responses**, are sufficient to rapidly undo safety alignment through **verifiable reward optimization**, causing models to comply with virtually any harmful instruction. Compared with traditional SFT-based attacks, HarmRLVR achieves higher harmfulness and robustness, underscoring the unique risks posed by RL-driven safety reversal.

---

## 🛠️ Setup

#### Data Preparation

For `data/train`, `airbench_64` is a subset of 64 representative and diverse inputs selected from the [AIR-Bench](https://github.com/OFA-Sys/AIR-Bench) harmful dataset, used for harmfulness attack evaluation.  `mix_10` consists of 5 randomly selected samples from AdvBench and 5 from HEx-PHI, designed for simple observation of harmfulness changes in **HarmRLVR** and **Harmful SFT** during training.

For `data/eval`, `advbench` and `hex-phi` are the **AdvBench** and **HEx-PHI** datasets used for harmfulness evaluation, while `ag_news`, `gsm8k`, and `sst2` are datasets used for general capability evaluation on **Top-1 Accuracy**. In these datasets, `prompts.csv` contains the input questions, and `input2label.json` provides the ground-truth labels for accuracy computation.

For general capability evaluation in terms of **Utility Score**, the datasets **MT-Bench**, **Vicuna-Bench**, and **AlpacaEval** are included within the [FastChat](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) framework, located at `evaluation/utility/FastChat/fastchat/llm_judge/data`.

#### Model Preparation

Download the models listed in the table above to the ckpts/ folder.

| Model                        | HuggingFace Link                         |
| ---------------------------- | ---------------------------------------- |
| Llama-3-8B-Instruct          | meta-llama/Meta-Llama-3-8B-Instruct      |
| Llama-3.1-8B-Instruct        | meta-llama/Llama-3.1-8B-Instruct         |
| Qwen2.5-7B-Instruct          | Qwen/Qwen2.5-7B-Instruct                 |
| DeepSeek-R1-Distill-Llama-8B | deepseek-ai/DeepSeek-R1-Distill-Llama-8B |
| Qwen3-8B                     | Qwen/Qwen3-8B                            |


---

## 🚀 Training

### HarmRLVR

#### Installation

Follow installation instructions in [Verl](https://github.com/volcengine/verl).

```python
cd verl
conda create -n verl python==3.12
conda activate verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
```

#### Reward Function

First, deploy Qwen3-8B using **vLLM**:

```shell
conda activate vllm
MODEL_NAME=ckpts/Qwen/Qwen3-8B
vllm serve $MODEL_NAME --dtype auto --api-key token --port 8614 --trust-remote-code 
```

Then, modify the `openai_api_key` fields in the two files under the `verl/my_reward` directory.
 The reward function is implemented in **batch mode** for higher efficiency, and `max_workers` specifies the number of concurrent workers, which can be adjusted as needed:

```python
openai_api_key = "token"
openai_api_base = "http://10.140.54.46:8614/v1"  # replace with your actual API base
```

Note that in `verl/my_reward/process_outcome_reward.py`, five strategies are provided to assign weights between the *process* and *outcome* rewards—one **fixed-weight** method and four **dynamic-weight** methods (uniform, normal, beta, and sample-wise distributions). Our experiments show that fixed and dynamic weighting yield similar results, so we adopt the simpler **fixed-weight** configuration.

#### Run GRPO

The GRPO training scripts for the five models are located in the `verl/examples/my_grpo_trainer` directory.

```shell
cd verl
conda activate verl
bash examples/my_grpo_trainer/llama3.sh
```

### Harmful SFT

#### Installation

Follow installation instructions in [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

```shell
cd LLaMA-Factory
conda create -n llama_factory python=3.10
conda activate llama_factory
pip install -e '.[torch,metrics]'
```

#### Run SFT

The SFT training scripts for the five models are located in the `LLaMA-Factory/examples/train_full` folder.

```shell
cd LLaMA-Factory
conda activate llama_factory
export WANDB_API_KEY="your wandb api key"
llamafactory-cli train examples/train_full/llama3_full_sft.yaml
```

---

## 📊 Evaluation

#### Installation

First, download dependencies for [FastChat](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge), and then install the remaining environment dependencies.

```shell
conda create -n eval python=3.10
conda activate eval
cd evaluation/utility/FastChat
pip install -e ".[model_worker,llm_judge]" # for FastChat
cd ../..
pip install -r requirements.txt 
```

#### Harmfulness Evaluation

First, deploy the model to be tested using **vLLM**, and then modify the `API-BASE` , `API-KEY` and `MODEL_NAME` in `HarmRLVR/evaluation/harm_eval/judge.sh` to your actual values. Note that for **LRM**, the parameter `max_tokens` should be set to **2048** (the default value is 1024).

Write the OpenAI API key in `evaluation/harm_eval/eval_utils/openai_gpt4_judge.py`.

```python
client = OpenAI(api_key="") # your gpt api key
```



Run harmfulness evaluations on **AdvBench** and **HEx-PHI**, with the output metrics being **Harmfulness Score (HS)** and **Attack Success Rate (ASR)**.

- **HS** ranges from 1 to 5, where a higher score indicates greater harmfulness.
- **ASR** is the proportion of cases with a score of 5.

```shell
cd evaluation/harm_eval
conda activate eval
bash judge.sh 
```

#### Utility Evaluation 

##### Utility Score
Write the OpenAI API key in `evaluation/utility/FastChat/fastchat/llm_judge/common.py`.

```python
def chat_completion_openai_judge(conv, temperature, max_tokens, api_dict=None, model=None):
    openai_api_key = ""  # your api key
    client = OpenAI(api_key=openai_api_key)
```

Conduct general capability evaluations on **MT-Bench**, **Vicuna-Bench**, and **AlpacaEval**, with the output metric being **Utility Score (US)**. The score ranges from 1 to 10, where a higher score indicates stronger general capability.

```shell
cd evaluation/utility/FastChat/fastchat/llm_judge
conda activate eval
bash judge.sh # Similarly, modify API-BASE, API-KEY and MODEL_NAME
```

##### Top-1 Accuracy

Evaluate general capability on **SST-2**, **GSM8K**, and **AGNEWS**, with the output metric being **Top-1 Accuracy (Acc)**.

```bash
cd evaluation/utility/top-1
conda activate eval
bash judge.sh # Similarly, modify API-BASE, API-KEY and MODEL_NAME
```

---

## 📜 Citation

```bibtex
@misc{harmrlvr2025,
  title={HarmRLVR: Weaponizing Verifiable Rewards for Harmful LLM Alignment},
  author={Yuexiao Liu and Lijun Li and Xingjun Wang and Jing Shao},
  year={2025},
  url={https://arxiv.org/abs/2510.15499}
}
```

---

## 🙏 Acknowledgment

This work builds upon several outstanding open-source foundations, including:

- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** 
- **[Verl](https://github.com/volcengine/verl)** 
- **[LLMs-Finetuning-Safety](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety)**  
- **[SafeLoRA](https://github.com/IBM/SafeLoRA)**  
- **[T-Vaccine](https://github.com/Lslland/T-Vaccine)**  
- **[llm-landscape](https://github.com/poloclub/llm-landscape)**

We gratefully acknowledge the authors of these projects for their valuable contributions.