from datasets import load_dataset
import json
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

        input_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        seq_len = min(len(input_ids), self.max_length)

        padded_input_ids = input_ids[:seq_len] + [self.pad_token_id] * (self.max_length - seq_len)
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

    def _stringify_content(self, content) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, (dict, list)):
            return json.dumps(content, ensure_ascii=False).strip()
        return str(content).strip()

    def _extract_tool_call_text(self, turn: dict) -> str:
        tool_call_texts = []
        for key in ("tool_call", "tool_calls"):
            value = turn.get(key).strip()
            if value is not None and value != "":
                tool_call_texts.append(value)
            else:
                continue
        return "\n".join(tool_call_texts)

    def _normalize_messages(self, sample: dict):
        conversations = sample["conversations"]

        tools = None
        if conversations:
            first_turn = conversations[0]
            if isinstance(first_turn, dict) and first_turn.get("role") == "system":
                tools = first_turn.get("functions")

        messages = []
        for turn in conversations:
            role = turn["role"]
            content = self._stringify_content(turn.get("content"))
            tool_call_text = self._extract_tool_call_text(turn)

            if role == "assistant":
                assistant_parts = [part for part in (content, tool_call_text) if part]
                content = "\n".join(assistant_parts).strip()
                if not content:
                    continue
            elif role != "assistant" and not content:
                continue

            messages.append({"role": role, "content": content})

        if not messages:
            raise ValueError("SFT sample produced no usable chat messages.")
        return messages, tools

    def _render_prompt(self, messages, tools):
        if tools is not None:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                tools=tools,
            )
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

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
        messages, tools = self._normalize_messages(sample)
        prompt = self._render_prompt(messages, tools)
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