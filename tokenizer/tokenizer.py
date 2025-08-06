class CharTokenizer:
    def __init__(self, text):
        self.sorted_chars = sorted(set(text))
        self.char_to_index = {c: i for i, c in enumerate(self.sorted_chars)}
        self.index_to_char = {i: c for c,i in self.char_to_index.items()}
        self.vocab_size = len(self.sorted_chars)
    
    def encoded(self, text):
        return [self.char_to_index[c] for c in text]
    
    def decoded(self, token_ids):
        return ''.join([self.index_to_char[i] for i in token_ids])
    
if __name__ == "__main__":
        text = "hello world"
        tokenizer = CharTokenizer(text)

        encoded = tokenizer.encoded("hello")
        print("Encoded:", encoded)

        decoded = tokenizer.decoded(encoded)
        print("Decoded:", decoded)
