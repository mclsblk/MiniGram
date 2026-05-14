from dataset.data_utils import PretrainDataset
from transformers import AutoTokenizer
from collections import Counter

tokenizer = AutoTokenizer.from_pretrained("./model")

print("Tokenizer loaded successfully.")
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
print(f"Tokenizer special tokens: {tokenizer.special_tokens_map}")

dataset = PretrainDataset(
    data_path="./dataset/pretrain_t2t.jsonl",
    tokenizer=tokenizer,
    max_length=340,
)

print("special_tokens_map_if_changed:", dataset.tokenizer.special_tokens_map)
print("Dataset loaded successfully.")

print("=" * 40)
# print("Sample Dataset Item:")
# sample_item = dataset[0]
# print(f"input_ids: {sample_item[0]}, length: {len(sample_item[0])}")
# print(f"labels: {sample_item[1]}, length: {len(sample_item[1])}")

counter = Counter()
for i in range(10000):
    input_ids, _ = dataset[i]
    counter.update(input_ids.tolist())
print("Most common tokens:", counter.most_common(10))
print("Freq of 201:", counter[201])
print("Freq of 223:", counter[223])
print("percentage of 201:", counter[0] / sum(counter.values()) * 100)
