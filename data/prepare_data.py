from tokenizer.tokenizer import CharTokenizer

file_path = "data/raw/input.txt"

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

print("First 100 Chars:\n", text[:100])

tokenizer = CharTokenizer(text)

data = tokenizer.encoded(text)