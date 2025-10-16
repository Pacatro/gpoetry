from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field


class TokenizerType(Enum):
    WORD = "word"
    CHAR = "char"


@dataclass
class TokenizerConfig:
    tk_type: str = TokenizerType.CHAR.value
    is_fitted: bool = False
    stoi: dict[str, int] = field(default_factory=dict)
    itos: dict[int, str] = field(default_factory=dict)
    vocab: list[str] = field(default_factory=list)
    vocab_size: int = 0


class Tokenizer(ABC):
    def __init__(self, config: TokenizerConfig | None = None):
        self.config = config or TokenizerConfig()

    @abstractmethod
    def _tokenize(self, text: str) -> list[str]:
        raise NotImplementedError("Tokenize method is not implemented")

    @abstractmethod
    def _detokenize(self, tokens: list[str]) -> str:
        raise NotImplementedError("Detokenize method is not implemented")

    def fit(self, texts: list[str] | str) -> None:
        if self.config.is_fitted:
            raise RuntimeError("Tokenizer is already fitted")

        all_tokens = set()

        if isinstance(texts, str):
            tokens = self._tokenize(texts)
            all_tokens.update(tokens)
        else:
            for text in texts:
                if not text:
                    continue
                tokens = self._tokenize(text)
                all_tokens.update(tokens)

        self.config.vocab = sorted(all_tokens)
        self.config.vocab_size = len(self.config.vocab)
        self.config.stoi = {token: idx for idx, token in enumerate(self.config.vocab)}
        self.config.itos = {idx: token for idx, token in enumerate(self.config.vocab)}
        self.config.is_fitted = True

    def encode(self, text: str) -> list[int]:
        if not self.config.is_fitted:
            raise RuntimeError("Tokenizer is not fitted")

        tokens = self._tokenize(text)
        return [self.config.stoi[t] for t in tokens]

    def decode(self, tokens: list[int]) -> str:
        assert len(tokens) > 0, "Cannot decode empty tokens"

        if not self.config.is_fitted:
            raise RuntimeError("Tokenizer is not fitted")

        decoded_tokens = [self.config.itos[t] for t in tokens]
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
