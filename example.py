import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_tokens(model, tokenizer, tokens_path=""):
    new_tokens_weights = torch.load(tokens_path)
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

    return model, tokenizer

model_path = "path/to/Mistral-7B-Instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model, tokenizer = load_tokens(model, tokenizer, tokens_path="/path/to/mistral.7b.instruct.added_token_embeddings.pt")
model.cuda()
model.eval()
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

test_data = {
    "question": "who got the first nobel prize in physics", 
    "target": ["Wilhelm Conrad R\u00f6ntgen"], 
    "ret_passages": [
        "Nobel Prize in Physics Nobel Prize in Physics The Nobel Prize in Physics () is a yearly award given by the Royal Swedish Academy of Sciences for those who have made the most outstanding contributions for mankind in the field of physics. It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895 and awarded since 1901; the others being the Nobel Prize in Chemistry, Nobel Prize in Literature, Nobel Peace Prize, and Nobel Prize in Physiology or Medicine. The first Nobel Prize in Physics was awarded to physicist Wilhelm R\u00f6ntgen in recognition of the extraordinary services he", 
        "Nobel Prize in Physics receive a diploma, a medal and a document confirming the prize amount. Nobel Prize in Physics The Nobel Prize in Physics () is a yearly award given by the Royal Swedish Academy of Sciences for those who have made the most outstanding contributions for mankind in the field of physics. It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895 and awarded since 1901; the others being the Nobel Prize in Chemistry, Nobel Prize in Literature, Nobel Peace Prize, and Nobel Prize in Physiology or Medicine. The first Nobel Prize in Physics was", 
        "Wilhelm Ro\u0308ntgen Wilhelm R\u00f6ntgen Wilhelm Conrad R\u00f6ntgen (; ; 27 March 1845 \u2013 10 February 1923) was a German mechanical engineer and physicist, who, on 8 November 1895, produced and detected electromagnetic radiation in a wavelength range known as X-rays or R\u00f6ntgen rays, an achievement that earned him the first Nobel Prize in Physics in 1901. In honour of his accomplishments, in 2004 the International Union of Pure and Applied Chemistry (IUPAC) named element 111, roentgenium, a radioactive element with multiple unstable isotopes, after him. Born to a German father and a Dutch mother, R\u00f6ntgen attended high school in Utrecht, Netherlands. In", 
        ]
}

added_tokens = [f" [ref{i}]" for i in range(1, 51)]
added_tokens = "".join(added_tokens)
text = [f"[1] {test_data['ret_passages'][0]}\n[2] {test_data['ret_passages'][1]}\n[3] {test_data['ret_passages'][2]}\n\n{added_tokens}Question: {test_data['question']}\nAnswer:"]
text_encode = tokenizer(text, padding="longest", max_length=750, truncation=True, return_attention_mask=True, return_tensors="pt", add_special_tokens=True)
text_input_ids = text_encode.input_ids.cuda()
text_attention_mask = text_encode.attention_mask.cuda()
outputs = model.generate(
    input_ids=text_input_ids,
    attention_mask=text_attention_mask,
    max_new_tokens=50,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)
batch_out_sentences = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("with virtual tokens: ", batch_out_sentences)

text = [f"[1] {test_data['ret_passages'][0]}\n[2] {test_data['ret_passages'][1]}\n[3] {test_data['ret_passages'][2]}\n\nQuestion: {test_data['question']}\nAnswer:"]
text_encode = tokenizer(text, padding="longest", max_length=750, truncation=True, return_attention_mask=True, return_tensors="pt", add_special_tokens=True)
text_input_ids = text_encode.input_ids.cuda()
text_attention_mask = text_encode.attention_mask.cuda()
outputs = model.generate(
    input_ids=text_input_ids,
    attention_mask=text_attention_mask,
    max_new_tokens=50,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)
batch_out_sentences = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("without virtual tokens: ", batch_out_sentences)
