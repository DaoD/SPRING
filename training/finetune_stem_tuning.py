from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import PromptTuningInit, PromptTuningConfig, TaskType
from spring_tuning_wrapper import SpringTuningForCausalLM
import torch
import wandb
import argparse
from accelerate import Accelerator
from typing import Optional
from fine_tune_qa_dataset import FineTuningQADataset


def tokenize_batch_for_finetune(batch, tokenizer: Optional[AutoTokenizer] = None, max_length: int = 1024):
    # prompt, completion, reference
    input_output_texts = [sample["reference"] + sample["prompt"] + " " + sample["completion"] + " " + tokenizer.eos_token for sample in batch]
    completion = [sample["completion"] + " " + tokenizer.eos_token for sample in batch]
    spring_insert_text = [sample["prompt"] + " " + sample["completion"] + " " + tokenizer.eos_token for sample in batch]
    data = tokenizer(input_output_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length, add_special_tokens=True)
    data_completion = tokenizer(completion, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length, add_special_tokens=False)
    data_spring_insert_text = tokenizer(spring_insert_text, padding=False, truncation=True, max_length=max_length, add_special_tokens=False)
    len_spring_insert_text = [len(data) for data in data_spring_insert_text["input_ids"]]
    data_mask_reverse = 1 - data_completion["attention_mask"]
    data_mask = data_mask_reverse * -100
    data["labels"] = data["input_ids"].clone()
    data["labels"] *= data_completion["attention_mask"]
    data["labels"] += data_mask
    data["insert_position"] = max_length - torch.tensor(len_spring_insert_text)
    data = {k: v.cuda() for k, v in data.items()}
    return data

def set_tokenizer(tokenizer):
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    return tokenizer

def run_ft(args):
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer = set_tokenizer(tokenizer)
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=50,
        prompt_tuning_init_text="According to the previous relevant passages, please answer the following question. Only return the answer without any other words.",
        tokenizer_name_or_path=args.tokenizer_path,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
    model = SpringTuningForCausalLM(model, peft_config)
    model.to(accelerator.device)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_path,
        learning_rate=args.learning_rate,
        num_train_epochs=3,
        per_device_train_batch_size=args.per_device_train_batch_size,
        dataloader_pin_memory=False,
        do_eval=False,
        save_total_limit=6,
        warmup_ratio=0.02,
        save_strategy="epoch",
        logging_steps=5,
        bf16=True,
        label_names=["completion"],
        remove_unused_columns=False,
        # gradient_checkpointing=True,
    )

    train_ds = FineTuningQADataset(args.dataset, with_ret=True, zero_shot=True, ret_passages=args.num_ret)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=lambda data: tokenize_batch_for_finetune(data, tokenizer=tokenizer, max_length=args.max_length)
    )

    trainer.train()
    trainer.save_model()
    trainer.save_state()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--max_length", type=int, default=600)
    parser.add_argument("--num_ret", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    args = parser.parse_args()
    wandb.init(mode="disabled")
    run_ft(args)

if __name__ == "__main__":
    main()