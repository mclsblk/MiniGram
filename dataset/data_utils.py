from datasets import load_dataset
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, text_field=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_field = text_field
        self.data = load_dataset("json", data_files=data_path, split="train")

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")
        if self.tokenizer.bos_token_id is None or self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define both bos_token_id and eos_token_id for pretraining.")

        self._text_fields = (
            [text_field]
            if text_field is not None
            else ["text", "content", "value", "prompt", "completion"]
        )

    def __len__(self):
        return len(self.data)

    def _get_text(self, sample):
        if isinstance(sample, str):
            return sample

        if not isinstance(sample, dict):
            return str(sample)

        for field in self._text_fields:
            if field in sample and sample[field] is not None:
                return str(sample[field])

        raise KeyError(
            f"Could not find a text field in dataset sample. Tried: {self._text_fields}. "
            f"Available keys: {list(sample.keys())}"
        )

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = self._get_text(sample)

        token_ids = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length - 2,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        bos = torch.tensor([self.tokenizer.bos_token_id], dtype=torch.long)
        eos = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
        input_ids = torch.cat([bos, token_ids, eos], dim=0)
        labels = input_ids.clone()

        padded_input_ids = torch.full((self.max_length,), self.tokenizer.pad_token_id, dtype=torch.long)
        padded_labels = torch.full((self.max_length,), -100, dtype=torch.long)

        seq_len = min(input_ids.size(0), self.max_length)
        padded_input_ids[:seq_len] = input_ids[:seq_len]
        padded_labels[:seq_len] = labels[:seq_len]

        return padded_input_ids, padded_labels
