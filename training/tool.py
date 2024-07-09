import torch
from transformers import AutoModelForCausalLM
from spring_tuning_wrapper import SpringTuningForCausalLM
from peft import PeftConfig


def get_new_embeddings(adapter_path: str = "", test_prompt_len = 50, embedding_save_path: str = None):
    peft_config = PeftConfig.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
    model = SpringTuningForCausalLM(model, peft_config, test_prompt_len=test_prompt_len)
    model.load_adapter(adapter_path, "default", is_trainable=False)
    new_embeddings = model.prompt_encoder.default.embedding.weight
    torch.save(new_embeddings, embedding_save_path)
    return new_tokens_and_embeddings

model_name_or_path = "path/to/output_path"
embedding_save_path = "path/to/save/embedding"
get_new_embeddings(model_name_or_path, embedding_save_path=embedding_save_path)
