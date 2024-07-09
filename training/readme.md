This is for training.


###

### Usage
```bash
TOKENIZERS_PARALLELISM=True accelerate launch --config_file zero1_gpu.yml finetune_spring_tuning.py \
  --model_path path/to/model \
  --output_path path/for/model/saving \
  --tokenizer_path path/to/tokenizer \
  --dataset path/to/data.jsonl \
  --max_length 800
```
