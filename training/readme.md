This is for training.


###

### Usage

### Required packages
```
accelerate            0.26.1
deepspeed             0.14.0
flash-attn            2.5.6
tqdm                  4.66.1
torch                 2.0.0
transformers          4.37.0
xformers              0.0.19
```

### Data format
The data should be saved in jsonl format, in which each line is a json dict as:
```python
{
    "question": String,
    "target": String or List[String],
    "ret_passages": List[String],
}
```
Below is a data example:
```python
test_data = {
    "question": "who got the first nobel prize in physics", 
    "target": ["Wilhelm Conrad R\u00f6ntgen"], 
    "ret_passages": [
        "Nobel Prize in Physics Nobel Prize in Physics The Nobel Prize in Physics () is a yearly award given by the Royal Swedish Academy of Sciences for those who have made the most outstanding contributions for mankind in the field of physics. It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895 and awarded since 1901; the others being the Nobel Prize in Chemistry, Nobel Prize in Literature, Nobel Peace Prize, and Nobel Prize in Physiology or Medicine. The first Nobel Prize in Physics was awarded to physicist Wilhelm R\u00f6ntgen in recognition of the extraordinary services he", 
        "Nobel Prize in Physics receive a diploma, a medal and a document confirming the prize amount. Nobel Prize in Physics The Nobel Prize in Physics () is a yearly award given by the Royal Swedish Academy of Sciences for those who have made the most outstanding contributions for mankind in the field of physics. It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895 and awarded since 1901; the others being the Nobel Prize in Chemistry, Nobel Prize in Literature, Nobel Peace Prize, and Nobel Prize in Physiology or Medicine. The first Nobel Prize in Physics was", 
        "Wilhelm Ro\u0308ntgen Wilhelm R\u00f6ntgen Wilhelm Conrad R\u00f6ntgen (; ; 27 March 1845 \u2013 10 February 1923) was a German mechanical engineer and physicist, who, on 8 November 1895, produced and detected electromagnetic radiation in a wavelength range known as X-rays or R\u00f6ntgen rays, an achievement that earned him the first Nobel Prize in Physics in 1901. In honour of his accomplishments, in 2004 the International Union of Pure and Applied Chemistry (IUPAC) named element 111, roentgenium, a radioactive element with multiple unstable isotopes, after him. Born to a German father and a Dutch mother, R\u00f6ntgen attended high school in Utrecht, Netherlands. In", 
        ]
}
```

### Training
Set the path and run the following code:
```bash
TOKENIZERS_PARALLELISM=True accelerate launch --config_file zero1_gpu.yml finetune_spring_tuning.py \
  --model_path path/to/model \
  --output_path path/for/model/saving \
  --tokenizer_path path/to/tokenizer \
  --dataset path/to/data.jsonl \
  --max_length 800
```

### Extract and Save Token Embeddings
Set the path and run the following code:
```bash
python tool.py
```
