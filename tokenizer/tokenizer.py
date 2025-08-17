import re

class Tokenizer:
    def __init__(self):
        self.encoder = {i: i for i in range(256)}
        self.decoder = {i: bytes([i]) for i in range(256)}
        self.special = {"<bos>": 256, "<eos>": 257, "<pad>": 258}
        self.encoder.update({k: v for k, v in self.special.items()})
        self.decoder.update({v:k.encode() for k, v in self.special.items()})
        self.vocab_size = 259

    def encode(self, text, add_special_tokens=True):
        b = text.encode("utf-8", errors="replace")
        ids = [self.special["<bos>"]] if add_special_tokens else []
        ids += list(b)
        if add_special_tokens:
            ids.append(self.special["<eos>"])
        return ids
    
    def decode(self, ids):
        out = bytearray()
        for i in ids:
            if i in self.special.values():
                continue
            if 0 <= i < 256:
                out.append(i)
        return out.decode("utf-8", errors="replace")