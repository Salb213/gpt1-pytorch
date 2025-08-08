import re
import numpy as np
from collections import Counter

class Tokenizer:
    def __init__(self, text, vocab_size=1000):
        vocab = Counter()
        for word in text.split():
            vocab[" ".join(list(word)) + " </w>"] += 1

        self.vocab = vocab


        self.merges = []
        while len(self.get_symbols()) < vocab_size:
            pairs = self.get_stats()
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            self.merge_vocab(best)

        symbols = self.get_symbols()
        self.encoder = {sym: i for i, sym in enumerate(symbols)}
        self.decoder = {i: sym for sym, i in self.encoder.items()}
        
        special_tokens = ["<bos>", "<eos>", "<pad>"]
        for tok in special_tokens:
            if tok not in self.encoder:
                idx = len(self.encoder)
                self.encoder[tok] = idx
                self.decoder[idx] = tok

        self.vocab_size = len(self.encoder)

    def get_stats(self):
        pairs = Counter()
        for word, freq in self.vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_vocab(self, pair):
        pattern = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + pattern + r"(?!\S)")
        new_vocab = {}
        for word, freq in self.vocab.items():
            new_word = pattern.sub("".join(pair), word)
            new_vocab[new_word] = freq
        self.vocab = new_vocab

    def get_symbols(self):
        symbols = set()
        for word in self.vocab:
            symbols.update(word.split())
        return symbols

    def encode(self, text, add_special_tokens=True):
        tokens = []
        if add_special_tokens:
            tokens.append(self.encoder["<bos>"]) 
        for word in text.split():
            chars = list(word) + ["</w>"]
            i = 0
            while i < len(chars):
                j = i + 1
                while j <= len(chars) and "".join(chars[i:j]) in self.encoder:
                    j += 1
                tokens.append(self.encoder["".join(chars[i:j-1])])
                i = j - 1
        if add_special_tokens:
            tokens.append(self.encoder["<eos>"])
        return tokens

    def decode(self, token_ids):
        parts = []
        for idx in token_ids:
            sym = self.decoder[idx]
            if sym == "<bos>" or sym == "<pad>":
                continue
            if sym == "<eos>":
                break
            if sym.endswith("</w>"):
                parts.append(sym[:-4] + " ")
            else:
                parts.append(sym)
        return "".join(parts).strip()

