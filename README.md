# SPRING: Learning Scalable and Pluggable Virtual Tokens for Retrieval-Augmented Large Language Models

<p>
<a href="https://github.com/DaoD/SPRING/blob/main/LICENSE"><img src="https://img.shields.io/badge/MIT-License-blue" alt="license"></a>
<a href="https://arxiv.org/abs/2405.19670"><img src="https://img.shields.io/badge/Paper-Arxiv-red"></a>
<a href="https://huggingface.co/yutaozhu94/SPRING"><img src="https://img.shields.io/badge/Embeddings-%F0%9F%A4%97%20Hugging%20Face-8A2BE2"></a>
</p>

**Authors**: Yutao Zhu, Zhaoheng Huang, Zhicheng Dou, and Ji-Rong Wen

<p>
🤗 HuggingFace Model List
</p>

| Model                                                                            | Backbone Model                                                          |
|:---------------------------------------------------------------------------------|:------------------------------------------------------------------------|
| [SPRING-LLaMA-2-7b-chat]()   | [LLaMA-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| [SPRING-LLaMA-2-7b-base]()   | [LLaMA-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf)           |
| [SPRING-Mistral-7b-instruct]()         | [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)          |
| [SPRING-Mistral-7b]()           | [Mistral-7b-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)               |
| [SPRING-Phi-3-mini]()           | [Phi-3-mini-4k-instruct](huggingface.co/microsoft/Phi-3-mini-4k-instruct)              |
| [SPRING-QWen-1.8b-chat]()           | [Qwen1.5-1.8B-Chat](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat)              |

## News
- May, 2024: We have released our paper on arXiv. The code and models are preparing and will be released later.

## Introduction

Retrieval-augmented generation (RAG) is a promising way to improve large language models (LLMs) for generating more factual, accurate, and up-to-date content. Existing methods either optimize prompts to guide LLMs in leveraging retrieved information or directly fine-tune the LLMs to adapt to RAG scenarios. Although fine-tuning can yield better performance, it often compromises the LLMs' general generation capabilities by modifying their parameters. This limitation poses challenges in practical applications, especially when LLMs are already deployed, as parameter adjustments may affect their original functionality. To address this, we propose a novel method that involves learning scalable and pluggable virtual tokens for RAG. By maintaining the LLMs’ original parameters and fine-tuning only the embeddings of these pluggable tokens, our approach not only enhances LLMs’ performance but also preserves their general generation capacities. Furthermore, we design several training strategies to improve the scalability, flexibility, and generalizability of our method. Comprehensive experiments across nine question-answering tasks demonstrate the superiority of our approach.

## Citation
Please kindly cite our paper if it helps your research:
```BibTex
@article{SPRING,
  author       = {Yutao Zhu and
                  Zhaoheng Huang and
                  Zhicheng Dou and
                  Ji{-}Rong Wen},
  title        = {One Token Can Help! Learning Scalable and Pluggable Virtual Tokens for Retrieval-Augmented Large Language Models},
  journal      = {CoRR},
  volume       = {abs/2405.19670},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2405.19670},
  doi          = {10.48550/ARXIV.2405.19670},
  eprinttype    = {arXiv},
  eprint       = {2405.19670}
}
```
