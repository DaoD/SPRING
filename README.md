# SPRING: Learning Scalable and Pluggable Virtual Tokens for Retrieval-Augmented Large Language Models

<p>
<a href="https://github.com/DaoD/SPRING/blob/main/LICENSE">
<img src="https://img.shields.io/badge/MIT-License-blue" alt="license">
</a>
</p>

**Authors**: Yutao Zhu, Zhaoheng Huang, Zhicheng Dou, and Ji-Rong Wen

<p>
📃 <a href="">ArXiv Paper</a>
  •
🤗 <a href="https://huggingface.co/">Dataset</a>
</p>

## Introduction

Retrieval-augmented generation (RAG) is a promising way to improve large language models (LLMs) for generating more factual, accurate, and up-to-date content. Existing methods either optimize prompts to guide LLMs in leveraging retrieved information or directly fine-tune the LLMs to adapt to RAG scenarios. Although fine-tuning can yield better performance, it often compromises the LLMs' general generation capabilities by modifying their parameters. This limitation poses challenges in practical applications, especially when LLMs are already deployed, as parameter adjustments may affect their original functionality. To address this, we propose a novel method that involves learning scalable and pluggable virtual tokens for RAG. By maintaining the LLMs’ original parameters and fine-tuning only the embeddings of these pluggable tokens, our approach not only enhances LLMs’ performance but also preserves their general generation capacities. Furthermore, we design several training strategies to improve the scalability, flexibility, and generalizability of our method. Comprehensive experiments across nine question-answering tasks demonstrate the superiority of our approach.

## Citation
Please kindly cite our paper if it helps your research:
```BibTex

}
```
