from enum import Enum


class Tokenization(Enum):
    WORD = "word"
    CHAR = "char"
    BPE = "bpe"


class Tokenizer:
    def __init__(self, mode: Tokenization):
        self._mode = mode
        self._stoi = None
        self._itos = None
        self.vocab = None
        self.vocab_size = 0

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        match self._mode:
            case Tokenization.WORD:
                tokens = text.split()
                self.vocab = sorted(set(tokens))
            case Tokenization.CHAR:
                tokens = list(text)
                self.vocab = sorted(set(tokens))
            case Tokenization.BPE:
                raise NotImplementedError("BPE tokenization is not implemented yet")

        self.vocab_size = len(self.vocab)
        self._stoi = {token: idx for idx, token in enumerate(self.vocab)}
        self._itos = {idx: token for idx, token in enumerate(self.vocab)}

        return [self._stoi[token] for token in tokens]

    def decode(self, tokens: list[int]) -> str:
        if not tokens:
            return ""

        assert self._itos is not None, "You must encode text before decoding"

        decoded_tokens = [self._itos[idx] for idx in tokens]

        match self._mode:
            case Tokenization.WORD:
                return " ".join(decoded_tokens)
            case Tokenization.CHAR:
                return "".join(decoded_tokens)
            case Tokenization.BPE:
                raise NotImplementedError("BPE tokenization is not implemented yet")
