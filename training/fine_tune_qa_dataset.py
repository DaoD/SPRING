import linecache
import json
import random
from torch.utils.data import Dataset


class FineTuningQADataset(Dataset):
    def __init__(self, filename, with_ret=False, zero_shot=True, ret_passages=10):
        super(FineTuningQADataset, self).__init__()
        self._filename = filename
        self._with_ret = with_ret
        self._zero_shot = zero_shot
        self._num_ret_passages = ret_passages
        with open(filename, "r", encoding="utf-8") as f:
            self._total_data = len(f.readlines())
    
    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        sample = json.loads(line)
        ret_passages = sample["ret_passages"]
        ret_passages = self._gen_ret_results(ret_passages)
        question_prompt = self._gen_prompt(sample)
        
        if isinstance(sample["target"], list):
            target = random.choice(sample["target"])
        else:
            target = sample["target"]

        batch = {
            "reference": ret_passages,
            "prompt": question_prompt,
            "completion": target  
        }

        return batch

    def __len__(self):
        return self._total_data

    def _gen_prompt(self, example, include_answer=False):
        prompt = f"Question: {example['question']}\n"  # prompt
        prompt += f"Answer:" # SPRING is free to add this or not
        return prompt

    def _gen_ret_results(self, ret_passages):
        # prompt = "According to the relevant passages below, please answer the following question.\n\n"
        prompt = ""
        _num_ret_passages = self._num_ret_passages
        if self._num_ret_passages == -1:
            _num_ret_passages = random.randint(1, 5)
        random.shuffle(ret_passages)
        for idx, p in enumerate(ret_passages[:_num_ret_passages]):
            prompt += f"[{idx + 1}] {p}\n"
        prompt += "\n"
        return prompt