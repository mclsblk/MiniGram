from datasets import load_dataset
import torch
from torch.utils.data import Dataset
import os
from dataset.chat_utils import normalize_conversations, render_chat_prompt
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_dataset("json", data_files=data_path, split="train")

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")
        if self.tokenizer.bos_token_id is None or self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define both bos_token_id and eos_token_id for pretraining.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample['text']

        token_ids = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length - 2,
            add_special_tokens=False,
        ).input_ids

        input_ids = [self.tokenizer.bos_token_id] + token_ids + [self.tokenizer.eos_token_id]
        seq_len = min(len(input_ids), self.max_length)

        padded_input_ids = input_ids[:seq_len] + [self.tokenizer.pad_token_id] * (self.max_length - seq_len)
        padded_labels = input_ids[:seq_len] + [-100] * (self.max_length - seq_len)

        return torch.tensor(padded_input_ids, dtype=torch.long), torch.tensor(padded_labels, dtype=torch.long)


class SFTDataset(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        max_length=512,
        train_on_prompt=False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_on_prompt = bool(train_on_prompt)
        self.data = load_dataset("json", data_files=data_path, split="train")

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")
        if self.tokenizer.bos_token is None or self.tokenizer.eos_token is None:
            raise ValueError("Tokenizer must define both bos_token and eos_token for SFT.")

        self.bos_id = self.tokenizer(
            f"{self.tokenizer.bos_token}assistant\n",
            add_special_tokens=False,
        ).input_ids
        self.eos_id = self.tokenizer(
            f"{self.tokenizer.eos_token}\n",
            add_special_tokens=False,
        ).input_ids
        self.max_skip_attempts = 5

    def __len__(self):
        return len(self.data)

    def _generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def _build_example(self, sample: dict):
        messages, tools = normalize_conversations(sample)
        prompt = render_chat_prompt(
            self.tokenizer,
            messages,
            tools=tools,
            add_generation_prompt=False,
        )
        input_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids[: self.max_length]
        if not input_ids:
            raise ValueError("SFT sample produced empty tokenized input.")

        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self._generate_labels(input_ids)
        if all(label == -100 for label in labels):
            raise ValueError("SFT sample has no supervised tokens after formatting/truncation.")
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        last_error = None
        for offset in range(self.max_skip_attempts):
            sample_idx = (idx + offset) % len(self.data)
            sample = self.data[sample_idx]
            try:
                return self._build_example(sample)
            except Exception as exc:
                last_error = exc

        raise ValueError(
            f"Failed to build a valid SFT sample after {self.max_skip_attempts} attempts "
            f"starting from index {idx}. Last error: {last_error}"
        )


class GRPODataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        conversations = sample["conversations"]
        reference = ""
        prompt_conversations = conversations
        if conversations and conversations[-1].get("role") == "assistant":
            reference = conversations[-1].get("content") or ""
            prompt_conversations = conversations[:-1]

        messages, tools = normalize_conversations(prompt_conversations)
        prompt = render_chat_prompt(
            self.tokenizer,
            messages,
            tools=tools,
            add_generation_prompt=True,
        )
        return {"prompt": prompt, "reference": reference}
