from tokenizers import Tokenizer

class ParakeetEouTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, path: str) -> 'ParakeetEouTokenizer':
        tokenizer = Tokenizer.from_file(f"{path}/tokenizer.json")
        return cls(tokenizer)

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)