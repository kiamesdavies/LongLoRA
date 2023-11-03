# Written by Yukang Chen
# Some code based on https://github.com/epfml/landmark-attention
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import os
import copy
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling
from llama_attn_replace_sft import replace_llama_attn
from gptneox_attn_replace import replace_gpt_neox_attn
from peft import LoraConfig, get_peft_model
from torch.distributed import barrier
import datasets

from concatenator import ConcatDataset

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


PROMPT_DICT = {
    "prompt_designer": (
        f"{B_INST} {B_SYS}Provided series of PromQL queries for several Grafana panels, generate a full Grafana dashboard.{E_SYS}```json\n{{designer_input}}\n```\n{E_INST}\n"
    ),
    "prompt_alchemist": (
        f"{B_INST} {B_SYS}Using the supplied Grafana dashboard graphs/panels in JSON – encompassing title, type, description, and associated metrics – and optionally the header of its associated group and a general dashboard summary, generate valid PromQL query.{E_SYS}```json\n{{alchemist_input}}\n```\n{E_INST}\n"
    )
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="EleutherAI/pythia-1.4b-deduped")
    model_type: Optional[str] = field(default="llama")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={
                           "help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    report_to: str = field(default="wandb")
    model_max_length: int = field(
        default=8192 * 4,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    use_full_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use plain, full-attention for training."},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    trainable_params: str = field(
        default="embed,norm",
        metadata={
            "help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )

def get_custom_dataset(tokenizer, split, data_path: str):
    dataset = datasets.load_dataset("kiamesdavies/prometheus-grafana-dashboards-full-v3",
                                    split=split)

    prompt_template = PROMPT_DICT["prompt_designer"] if data_path == "designer" else PROMPT_DICT["prompt_alchemist"]
    output_template = f"[RESULT]```json\n{{designer_output}}\n```[/RESULT]" if data_path == "designer" else f"[RESULT]```json\n{{alchemist_input}}\n```[/RESULT]"

    def apply_prompt_template(sample):
        return {
            "prompt": prompt_template.format(**sample),
            "output": output_template.format(**sample),
        }

    dataset = dataset.map(apply_prompt_template,
                          remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(
            tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(
            sample["output"] + tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + summary,
            "attention_mask": [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
        }

        return sample

    dataset = dataset.map(tokenize_add_label,
                          remove_columns=list(dataset.features))

    return dataset


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, training_args, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_train = get_custom_dataset(
        data_path=data_args.data_path, tokenizer=tokenizer, split="train")
    dataset_eval = get_custom_dataset(
        data_path=data_args.data_path, tokenizer=tokenizer, split="test")

    return dict(train_dataset=ConcatDataset(dataset_train, chunk_size=training_args.model_max_length), eval_dataset=dataset_eval)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # NOTE: May expand supported model types in the future
    if model_args.model_type == "gpt-neox":
        replace_gpt_neox_attn(training_args.use_flash_attn,
                              training_args.use_full_attn)
    else:
        replace_llama_attn(training_args.use_flash_attn,
                           training_args.use_full_attn)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(
            math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id


    data_module = make_supervised_data_module(
        tokenizer=tokenizer, training_args=training_args, data_args=data_args)

    if training_args.low_rank_training:
        if model_args.model_type == "gpt-neox":
            # added `dense` to match with llama as the basic LoRA would only target 'query_key_value'
            targets = ["query_key_value", "dense"]
        else:
            targets = ["q_proj", "k_proj", "v_proj", "o_proj"]

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=targets,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        # enable trainable params
        [p.requires_grad_() for n, p in model.named_parameters() if any(
            [k in n for k in training_args.trainable_params.split(",")])]

    model.config.use_cache = False         # required for gradient checkpointing
    model.enable_input_require_grads()     # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing

    trainer = Trainer(model=model, tokenizer=tokenizer,
                      args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
