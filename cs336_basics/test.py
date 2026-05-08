import json

with open("data/TinyStoriesV2-GPT4-train/vocab.json", "r", encoding="utf-8") as f:
    raw_vocab = json.load(f)

token_id, token = max(raw_vocab.items(), key=lambda item: len(item[1]))

print("token_id:", token_id)
print("string length:", len(token))
print("token:", repr(token))