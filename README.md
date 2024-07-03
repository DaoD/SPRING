# SPRING: Learning Scalable and Pluggable Virtual Tokens for Retrieval-Augmented Large Language Models

<p>
<a href="https://github.com/DaoD/SPRING/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue" alt="license"></a>
<a href="https://arxiv.org/abs/2405.19670"><img src="https://img.shields.io/badge/Paper-Arxiv-red"></a>
<a href="https://huggingface.co/yutaozhu94/SPRING"><img src="https://img.shields.io/badge/Embeddings-%F0%9F%A4%97%20Hugging%20Face-8A2BE2"></a>
</p>

**Authors**: Yutao Zhu, Zhaoheng Huang, Zhicheng Dou, and Ji-Rong Wen

| Virtual token embeddings file                                                    | Backbone Model                                                          |
|:---------------------------------------------------------------------------------|:------------------------------------------------------------------------|
| mistral.7b.instruct.added_token_embeddings.pt       | [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)                |
| mistral.7b.base.added_token_embeddings.pt           | [Mistral-7b-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)                                  |
| llama2.7b.chat.added_token_embeddings.pt            | [LLaMA-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)                              |
| llama2.7b.base.added_token_embeddings.pt            | [LLaMA-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf)                                        |
| llama2.13b.chat.added_token_embeddings.pt           | [LLaMA-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)                             |
| llama2.13b.base.added_token_embeddings.pt           | [LLaMA-2-7b-base](https://huggingface.co/meta-llama/Llama-2-13b-hf)                                  |

## News
- May, 2024: We have released our paper on arXiv. The code and models are preparing and will be released later.

## Introduction

Retrieval-augmented generation (RAG) is a promising way to improve large language models (LLMs) for generating more factual, accurate, and up-to-date content. Existing methods either optimize prompts to guide LLMs in leveraging retrieved information or directly fine-tune the LLMs to adapt to RAG scenarios. Although fine-tuning can yield better performance, it often compromises the LLMs' general generation capabilities by modifying their parameters. This limitation poses challenges in practical applications, especially when LLMs are already deployed, as parameter adjustments may affect their original functionality. To address this, we propose a novel method that involves learning scalable and pluggable virtual tokens for RAG. By maintaining the LLMs’ original parameters and fine-tuning only the embeddings of these pluggable tokens, our approach not only enhances LLMs’ performance but also preserves their general generation capacities. Furthermore, we design several training strategies to improve the scalability, flexibility, and generalizability of our method. Comprehensive experiments across nine question-answering tasks demonstrate the superiority of our approach.

## Usage

### Required packages
```
torch                 2.0.0
transformers          4.37.0
```

### Example code
We provide an example coed in `example.py`

### Load token embeddings
```python
def load_tokens(model, tokenizer, token_embedding_path=""):
    new_tokens_weights = torch.load(token_embedding_path)
    new_tokens_length = new_tokens_weights.shape[0]

    # expand vocabulary
    new_tokens = [f"[ref{i+1}]" for i in range(new_tokens_length)]
    tokenizer.add_tokens(new_tokens)
    
    # get original embedding weight matrix
    embedding_layer = model.get_input_embeddings()
    embedding_weights = embedding_layer.weight
    original_vocab_size, embedding_dim = embedding_weights.shape
    
    # create new embedding matrix
    new_vocab_size = original_vocab_size + new_tokens_length
    new_embedding_weights = torch.zeros(new_vocab_size, embedding_dim)

    # copy original embeddings to the new weights
    new_embedding_weights[:original_vocab_size, :] = embedding_weights

    # append virtual token embeddings to the new weights
    for token, embedding in zip(new_tokens, new_tokens_weights):
        token_id = tokenizer.convert_tokens_to_ids(token)
        new_embedding_weights[token_id] = embedding
    
    # update the embedding table
    # note: we should avoid using the function resize_token_embeddings() because this function will also change the lm_head of the model
    embedding_layer.weight.data = new_embedding_weights

    # model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

model_path = "path/to/Mistral-7B-Instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model, tokenizer = load_tokens(model, tokenizer, token_embedding_path="/path/to/mistral.7b.instruct.added_token_embeddings.pt")
```

### Add virtual tokens to the input
```python
# using 50 tokens as an example
added_tokens = [f" [ref{i}]" for i in range(1, 51)]
added_tokens = "".join(added_tokens)
retrieved_results = "..."
question = "..."
text = [f"{retrieved_results}{added_tokens}Question: {question}\nAnswer:"]

...

outputs = model.generate(...)

```


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
