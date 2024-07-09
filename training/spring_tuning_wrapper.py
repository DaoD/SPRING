import random
import torch
import warnings
import transformers
import packaging.version
from peft import PeftModel, PeftType, PeftConfig
from peft.utils import _get_batch_size
from typing import Optional


class SpringTuningForCausalLM(PeftModel):
    def __init__(self, model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = "default", test_prompt_len: int = 20) -> None:
        super().__init__(model, peft_config, adapter_name)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
        self.test_prompt_len = test_prompt_len

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        insert_position=None,
        **kwargs,
    ):
        batch_size = _get_batch_size(input_ids, inputs_embeds)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                # "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
                # "insert_position": insert_position,  # for inference
            }
        )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)  # b, n, d
        current_num_virtual_tokens = random.randint(1, 50)
        # current_num_virtual_tokens = 30
        if labels is not None:
            # stem_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
            stem_labels = torch.full((batch_size, current_num_virtual_tokens), -100).to(labels.device)
            kwargs["labels"] = torch.cat((stem_labels, labels), dim=1)
        prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)  # b, n, d
        prompts = prompts.to(inputs_embeds.dtype)
        prompts = prompts[:, :current_num_virtual_tokens, :]
        new_inputs_embeds = []
        new_attention_mask = []
        # prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)  # [bsz, n]
        prefix_attention_mask = torch.ones(batch_size, current_num_virtual_tokens).to(attention_mask.device)  # [bsz, n]
        for i in range(batch_size):
            lens = insert_position[i]
            inputs_embeds_left = inputs_embeds[i, :lens, :]  # n, d
            inputs_embeds_right = inputs_embeds[i, lens:, :]
            new_inputs_embeds.append(torch.cat([inputs_embeds_left, prompts[i], inputs_embeds_right], dim=0))
            attention_mask_left = attention_mask[i, :lens]  # n
            attention_mask_right = attention_mask[i, lens:]
            new_attention_mask.append(torch.cat([attention_mask_left, prefix_attention_mask[i], attention_mask_right], dim=0))
        inputs_embeds = torch.stack(new_inputs_embeds, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)
        kwargs.update(
            {
                "attention_mask": attention_mask,
            }
        )
        return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def generate(self, insert_position, *args, **kwargs):
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        kwargs["insert_position"] = insert_position
        if hasattr(self.base_model, "model"):
            self.base_model.model.generation_config = self.generation_config
        else:
            self.base_model.generation_config = self.generation_config
        try:
            outputs = self.base_model.generate(*args, **kwargs)
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            return outputs

    def prepare_inputs_for_generation(self, *args, task_ids: Optional[torch.Tensor] = None, insert_position=None, **kwargs):
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)

        # https://github.com/huggingface/transformers/pull/26681/ introduced new cache format
        # for some architectures which requires a special fix for prompt tuning etc.
        # TODO: starting with transformers 4.38, all architectures should support caching.
        uses_transformers_4_38 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.38.0")
        uses_transformers_4_36 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.36.0")
        transformers_new_cache_archs = ["llama", "mistral", "persimmon", "phi"]
        uses_cache = uses_transformers_4_38 or (
            uses_transformers_4_36 and self.base_model.config.model_type in transformers_new_cache_archs
        )

        if peft_config.is_prompt_learning:
            if uses_cache and (model_kwargs["past_key_values"] is not None):
                # change in the logic of `prepare_inputs_for_generation` makes the below code necessary
                # In prompt learning methods, past key values are longer when compared to the `input_ids`.
                # As such only consider the last input ids in the autogressive generation phase.
                if model_kwargs["past_key_values"][0][0].shape[-2] >= model_kwargs["input_ids"].shape[1]:
                    model_kwargs["input_ids"] = model_kwargs["input_ids"][:, -1:]

            if model_kwargs.get("attention_mask", None) is not None:
                # size = model_kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens
                size = (model_kwargs["input_ids"].shape[0], self.test_prompt_len)
                prefix_attention_mask = torch.ones(size).to(model_kwargs["input_ids"].device)
                new_attention_mask = []
                for i in range(size[0]):
                    lens = insert_position[i]
                    attention_mask_left = model_kwargs["attention_mask"][i, :lens]  # n
                    attention_mask_right = model_kwargs["attention_mask"][i, lens:]
                    new_attention_mask.append(torch.cat([attention_mask_left, prefix_attention_mask[i], attention_mask_right], dim=0))
                model_kwargs["attention_mask"] = torch.stack(new_attention_mask, dim=0)

            if model_kwargs.get("position_ids", None) is not None:
                warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
                model_kwargs["position_ids"] = None

            if kwargs.get("token_type_ids", None) is not None:
                warnings.warn(
                    "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                )
                kwargs["token_type_ids"] = None

            if model_kwargs["past_key_values"] is None:
                inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
                prompts = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0], task_ids=task_ids)
                prompts = prompts.to(inputs_embeds.dtype)
                prompts = prompts[:, :self.test_prompt_len, :]
                batch_size = model_kwargs["input_ids"].shape[0]
                new_inputs_embeds = []
                for i in range(batch_size):
                    lens = insert_position[i]
                    inputs_embeds_left = inputs_embeds[i, :lens, :]  # n, d
                    inputs_embeds_right = inputs_embeds[i, lens:, :]
                    new_inputs_embeds.append(torch.cat([inputs_embeds_left, prompts[i], inputs_embeds_right], dim=0))
                model_kwargs["inputs_embeds"] = torch.stack(new_inputs_embeds, dim=0)
                model_kwargs["input_ids"] = None

        # For transformers>=4.38.0 - for some architectures such as Llama, `cache_position` is
        # passed in the forward pass to keep track of the position ids of the cache. We have to
        # pop that from `model_kwargs` as `cache_position` is properly created by the model, using the passed
        # `inputs_embeds`: https://github.com/huggingface/transformers/blob/593230f0a1150ea9c0477b9d859f25daf73c8c33/src/transformers/models/llama/modeling_llama.py#L956
        _ = model_kwargs.pop("cache_position", None)

        return model_kwargs