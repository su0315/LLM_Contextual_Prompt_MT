# Exploring Context Representation for Machine Translation with Large Language Models

The repository containes the relevant code for the master thesis project "Exploring Context Representation for Machine Translation with Large Language Models".
The project is divided into three parts, and the following guide you how to replicate the result. 

1. Context-aware MT with LLMs 
2. Summarised-Context-aware MT with LLMs
3. Evaluating Discourse Phenomena on MT with LLMs

## Get started
1. Clone the repository
```bash
git clone https://github.com/su0315/LLM_Contextual_Prompt_MT.git
```
2. Clone [LTI's Texxt Generation Inference](https://github.com/CoderPat/text-generation-inference) for running the LLM.
```bash
git clone https://github.com/CoderPat/text-generation-inference.git
```
3. Change directory to the repository, and replicate the environment using ```llm_mt.yml```
```bash 
cd LLM_Contextual_Prompt_MT \
conda env create -f llm_mt.yml
```
4. Activate the environment
```bash
conda activate llm_mt
```
5. Move to the ```main``` directory.
```bash
cd main
```

## 1. Context-aware MT with LLMs
#### Context-agnostic Model
1. In ```eval_mt.sh```, replace ```tgt_langs```, ```context_sizes```, ```context_side``` of your interest.
For example, if you want to see Arabic, source-side context size 3:
```bash 
tgt_langs="ar" \
context_size=3 \
context_side="src"
```
2. And then you can run:
```bash
sh main/eval_mt.sh 
```
When you want to run context-agnostic MT model, set context size 0.

