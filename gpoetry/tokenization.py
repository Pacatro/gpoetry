from abc import ABC, abstractmethod
from enum import Enum


class TokenizerType(Enum):
    WORD = "word"
    CHAR = "char"


class Tokenizer(ABC):
    def __init__(self):
        self._is_fitted: bool = False
        self._stoi: dict[str, int] = {}
        self._itos: dict[int, str] = {}
        self.vocab: list[str] = []

    @abstractmethod
    def _tokenize(self, text: str) -> list[str]:
        raise NotImplementedError("Tokenize method is not implemented")

    @abstractmethod
    def _detokenize(self, tokens: list[str]) -> str:
        raise NotImplementedError("Detokenize method is not implemented")

    def fit(self, texts: list[str]) -> None:
        if self._is_fitted:
            raise RuntimeError("Tokenizer is already fitted")

        all_tokens = set()

        for text in texts:
            if not text:
                continue
            tokens = self._tokenize(text)
            all_tokens.update(tokens)

        self.vocab = sorted(all_tokens)
        self.vocab_size = len(self.vocab)
        self._stoi = {token: idx for idx, token in enumerate(self.vocab)}
        self._itos = {idx: token for idx, token in enumerate(self.vocab)}
        self._is_fitted = True

    def encode(self, text: str) -> list[int]:
        if not self._is_fitted:
            raise RuntimeError("Tokenizer is not fitted")

        tokens = self._tokenize(text)
        return [self._stoi[t] for t in tokens]

    def decode(self, tokens: list[int]) -> str:
        assert len(tokens) > 0, "Cannot decode empty tokens"

        if not self._is_fitted:
            raise RuntimeError("Tokenizer is not fitted")

        decoded_tokens = [self._itos[t] for t in tokens]
        return self._detokenize(decoded_tokens)


class WordTokenizer(Tokenizer):
    def _tokenize(self, text: str) -> list[str]:
        return text.split()

    def _detokenize(self, tokens: list[str]) -> str:
        return " ".join(tokens)


class CharTokenizer(Tokenizer):
    def _tokenize(self, text: str) -> list[str]:
        return list(text)

    def _detokenize(self, tokens: list[str]) -> str:
        return "".join(tokens)
