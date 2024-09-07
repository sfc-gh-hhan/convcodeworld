# ConvCodeWorld & ConvCodeBench
<center>
<img src="images/logo.png" alt="ConvCodeWorld">
</center>

<p align="center">
    <a href="https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard"><img src="https://img.shields.io/badge/🤗&nbsp&nbsp%F0%9F%8F%86-leaderboard-%23ff8811"></a>
    <a href="https://arxiv.org/abs/2406.15877"><img src="https://img.shields.io/badge/arXiv-2406.15877-b31b1b.svg"></a>
    <a href="https://pypi.org/project/bigcodebench/"><img src="https://img.shields.io/pypi/v/bigcodebench?color=g"></a>
    <a href="https://github.com/bigcodebench/bigcodebench/blob/master/LICENSE"><img src="https://img.shields.io/pypi/l/bigcodebench"></a>
</p>

<p align="center">
    <a href="#-about">🎙️About</a> •
    <a href="#-quick-start">🔥Quick Start</a> •
    <a href="#-full-script">🚀Full Script</a> •
    <a href="#-result-analysis">📊Result Analysis</a> •
    <a href="#-llm-generated-code">💻LLM-generated Code</a> •
    <a href="#-known-issues">🐞Known Issues</a> •
    <a href="#-citation">📜Citation</a> •
    <a href="#-acknowledgement">🙏Acknowledgement</a>
</p>

## News
- **[2024-09-XX]** We release ConvCodeWorld, a new benchmark for code generation with 1140 software-engineering-oriented programming tasks. Preprint is available [here](). PyPI package is available [here]() with the version `0.3.6`.

## 🎙️ About

<center>
<img src="images/ConvCodeWorld_ConvCodeBench.png" alt="ConvCodeWorld">
</center>

### ConvCodeWorld

ConvCodeWorld is a reproducible world that supports diverse feedback combination for conversational code generation.

### Why ConvCodeWorld?

ConvCodeWorld focuses on the evaluation of ...

### ConvCodeBench

ConvCodeBench is a cost-effective benchmark that strongly correlates to ConvCodeWorld.

### Why ConvCodeWorld?

ConvCodeBench is cheaper...


* ✨ **Precise evaluation & ranking**: See [our leaderboard]() for latest LLM rankings before & after rigorous evaluation.
* ✨ **Pre-generated samples**: ConvCodeWorld accelerates code intelligence research by open-sourcing [LLM-generated samples](#-LLM-generated-code) for various models -- no need to re-run the expensive benchmarks!

## 🔥 Quick Start

To get started, please first set up the environments:

### Install MiniConda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
conda install conda-forge::conda-ecosystem-user-package-isolation
# Restart the kernel
```

### Setup Conda Environments
All you need to do is run `setup.sh` 😊.
```bash
bash setup.sh
```
This script will create three environments:
- `ConvCodeWorld`: The main environment for ConvCode[World|Bench] 
- `bigcodebench`: [BigCodeBench](https://github.com/bigcode-project/bigcodebench) for initial code generation and code execution
- `vllm`: [vLLM](https://github.com/vllm-project/vllm) to accelerate inference speed of open-source LLMs

### OpenAI API Key 
Please save your API key in `convcodeworld` folder (where `run.py` is placed).
```bash
cd convcodeworld
echo $OPENAI_API_KEY > .api_key
```
This is necessary if 1) you run on ConvCodeWorld, or 2) you want to use OpenAI models for code generation.  

### Run vLLM 
If you want to use open-source models for code generation, you need to run: 
```bash
bash run_vllm.sh $MODEL_NAME
# Now open another kernel and run ConvCode[World|Bench]!  
```
Note that `$MODEL_NAME` is a full huggingface name such as `deepseek-ai/deepseek-coder-6.7b-instruct`. 
The default setting is to use `bfloat16` and to occupy a single GPU.
If you want to use quantization, you can simply include `--quantization="fp8"` in `run_vllm.sh`.
Similarly, if you want to use `n` gpus, you can include: `--tensor-parallel-size n`.


## 🚀 Full Script

### ConvCodeWorld
 To run ConvCodeWorld, we provide a sample script for the full pipeline: 
```bash
bash run_convcodeworld.sh $MODEL_NAME $EXECUTION_FEEDBACK $PARTIAL_TEST $SIMULATED_USER_FEEDBACK $USER_EXPERTISE
```
- `MODEL_NAME`: A full huggingface name such as `deepseek-ai/deepseek-coder-6.7b-instruct`.  
- `EXECUTION_FEEDBACK` (`true` or `false`): `true` if employ execution feedback. 
- `PARTIAL_TEST` (`true`, `false`, or `none`): `true` if test coverage is low (using only public test cases). `none` if `EXECUTION_FEEDBACK` is `false`. 
- `SIMULATED_USER_FEEDBACK` (`true` or `false`): `true` if employ user feedback simulation by GPT-4o. 
- `USER_EXPERTISE` (`novice`, `expert`, or `none`): User expertise for simulated user feedback. `none` if `SIMULATED_USER_FEEDBACK` is `false`. 

Note that compilation feedback is always included.

#### Example
If you want to run `deepseek-ai/deepseek-coder-6.7b-instruct` while feeding execution feedback with high test coverage and novice-level user feedback:
```bash
bash run_convcodeworld.sh deepseek-ai/deepseek-coder-6.7b-instruct true false true novice
```
 

### ConvCodeBench
To run ConvCodeBench, we also provide a sample script as follows: 

```bash
bash run_convcodebench.sh $MODEL_NAME $EXECUTION_FEEDBACK $PARTIAL_TEST $SIMULATED_USER_FEEDBACK $USER_EXPERTISE $REF_MODEL_NAME
```
- `MODEL_NAME`: A full huggingface name such as `deepseek-ai/deepseek-coder-6.7b-instruct`.  
- `EXECUTION_FEEDBACK` (`true` or `false`): `true` if employ execution feedback. 
- `PARTIAL_TEST` (`true`, `false`, or `none`): `true` if test coverage is low (using only public test cases). `none` if `EXECUTION_FEEDBACK` is `false`. 
- `SIMULATED_USER_FEEDBACK` (`true` or `false`): `true` if employ user feedback simulation by GPT-4o. 
- `USER_EXPERTISE` (`novice`, `expert`, or `none`): User expertise for simulated user feedback. `none` if `SIMULATED_USER_FEEDBACK` is `false`. 
- `REF_MODEL_NAME`: The reference model name. We recommend `deepseek-ai/deepseek-coder-6.7b-instruct`.  




## 📊 Result Analysis

We provide a script to replicate the analysis like MRR and C-Recall ...


To run the analysis, you need to ... 

```bash
python print_results.py ...
```


## 💻 LLM-generated Code

We share pre-generated code samples from LLMs we have [evaluated]().

## 🐞 Known Issues


## 📜 Citation

```bibtex

```

## 🙏 Acknowledgement

- [BigCodeBench](https://github.com/bigcode-project/bigcodebench)
- [DSPy](https://github.com/stanfordnlp/dspy)
- [vLLM](https://github.com/vllm-project/vllm)