# Exploring Context Representation for Machine Translation with Large Language Models

The repository containes the relevant code for the master thesis project "Exploring Context Representation for Machine Translation with Large Language Models".
The project is divided into three parts, and the following guide you how to replicate the result. 

1. Context-aware MT with LLMs 
2. Summarised-Context-aware MT with LLMs
3. Evaluating Discourse Phenomena on MT with LLMs

## Get started
1. Clone the repository.
```bash
git clone https://github.com/su0315/LLM_Contextual_Prompt_MT.git
```
2. Clone [LTI's Text Generation Inference](https://github.com/CoderPat/text-generation-inference) for running the LLM. Follow the setup instruction in the [README](https://github.com/CoderPat/text-generation-inference/blob/main/README.md).
```bash
git clone https://github.com/CoderPat/text-generation-inference.git
```
3. Change directory to the repository, and replicate the environment using ```llm_mt.yml```.
```bash 
cd LLM_Contextual_Prompt_MT \
conda env create -f llm_mt.yml
```
4. Activate the environment.
```bash
conda activate llm_mt
```
5. Download dataset. We use IWSLT17, which we downloaded from [here](https://huggingface.co/datasets/iwslt2017/tree/main/data/2017-01-trnted/texts).

6. Convert the XML dataset to json format for our experiment and prepare dataset by running:
```bash
cd data \
sh prepare_data.sh
```

5. Move to the ```main``` directory.
```bash
cd main
```

## 1. Context-aware MT with LLMs
#### Evaluate Context-agnostic / Context-aware Models
1. In ```eval_mt.sh```, replace ```tgt_langs```, ```context_sizes```, and ```context_side``` for your research interest.
For example, if you want to evaluate Arabic, source-side context size 3:
```bash 
tgt_langs="ar" \
context_size=3 \
context_side="src"
```
2. And then you can run:
```bash
sh eval_mt.sh 
```
When you want to run context-agnostic MT model, set context size 0.

3. To evaluate CXMI score (the impact of context), run:
```bash
sh cxmi.sh
```

## 2. Summarised-Context-aware MT with LLMs
#### Summarise context
1. We use [Transformersum](https://transformersum.readthedocs.io/en/latest/index.html) for summarising context. Follow [the instruction](https://transformersum.readthedocs.io/en/latest/general/getting-started.html).

2. In ```eval_summ.sh```, replace ```tgt_langs```, ```context_sizes```, ```context_side```, and ```summarized_sizes``` for your research interest. 
For example, if you want to summarise the source-side context with the target language German, Chinese, Arabic, Japanese, context size 8, the number of summarised sentences 2:
```bash
tgt_langs="de zh ar ja" 
context_sizes="8"
summarized_sizes="2"
context_side="src"
```
#### Evaluate summarised-context-aware MT
1.  In ```eval_summarized_mt.sh```, replace ```tgt_langs```, ```context_sizes```, ```context_side```, and ```summarized_sizes``` as above.
2. And then you can run:
```bash
sh eval_mt.sh 
```
3. To evaluate CXMI score (the impact of context), run:
```bash
sh summarized_cxmi.sh
```

## 3. Evaluating Discourse Phenomena on MT with LLMs
#### Discourse Phenomena Resolutions for lexical cohesion, formality, pronouns, and verb forms.
1. We use [MuDA](https://github.com/CoderPat/MuDA) for summarising context. Follow the [README](https://github.com/CoderPat/MuDA/blob/main/README.md) for the setup.
2. To run MuDA tagger on our dataset, in ```muda_eval.sh```, replace ```muda_lang``` and ```lang```.
For example if you want to run MuDA for Japanese, it looks like this, since the language code is slightly different each other (However, except for Japanese, you can write the same for both).
```bash
muda_lang=jp \
lang=ja \
```
3. Then, run the script.
```bash
sh muda_eval.sh
```
#### Evaluating MT quality of the instances with discourse phenomena
1. In ```eval_bleu_comet.sh```, replace ```criterias``` and ```tgt_langs``` for your research interest.
For example, if you want to evaluate the instances of lexical cohesion and formality of Korean, with the context size 1, 2, 3 and 4 for the target-side: 
```bash
criterias="lexical_cohesion formality" \
context_sizes="1-2 2-3 1-4 1-5" \
tgt_langs="ko" \
```
2. Run the script.
```bash
sh eval_bleu_comet.sh
```
#### Evaluate P-CXMI
1. In ```p-cxmi.sh```, replace ```langs``` and ```contexts``` for your researhc interest.
For example, if you want to evaluate Korean with the source-side context size 3:
```bash
langs="ko"
contexts="4-1"
```
2. Run the script.
```bash
sh p-cxmi.sh
```

