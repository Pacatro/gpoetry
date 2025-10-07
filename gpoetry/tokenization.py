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
        self._is_fitted = False
        self.vocab = None
        self.vocab_size = 0

    def fit(self, texts: list[str]) -> None:
        if self._is_fitted:
            raise RuntimeError("Tokenizer is already fitted")

        all_tokens = set()

        for text in texts:
            if not text:
                continue

            match self._mode:
                case Tokenization.WORD:
                    tokens = text.split()
                case Tokenization.CHAR:
                    tokens = list(text)
                case Tokenization.BPE:
                    raise NotImplementedError("BPE tokenization is not implemented yet")

            all_tokens.update(tokens)

        self.vocab = sorted(all_tokens)
        self.vocab_size = len(self.vocab)

        self._stoi = {token: idx for idx, token in enumerate(self.vocab)}
        self._itos = {idx: token for idx, token in enumerate(self.vocab)}

        self._is_fitted = True

    def encode(self, text: str) -> list[int]:
        if not self._is_fitted:
            raise RuntimeError("Tokenizer is not fitted")

        if not text:
            return []

        match self._mode:
            case Tokenization.WORD:
                tokens = text.split()
            case Tokenization.CHAR:
                tokens = list(text)
            case Tokenization.BPE:
                raise NotImplementedError("BPE tokenization is not implemented yet")

        assert self._stoi is not None and self._itos is not None

        return [self._stoi[token] for token in tokens]

    def decode(self, tokens: list[int]) -> str:
        if not tokens:
            return ""

        if not self._is_fitted:
            raise RuntimeError("Tokenizer is not fitted")

        assert self._itos is not None, "You must encode text before decoding"

        decoded_tokens = [self._itos[idx] for idx in tokens]

        match self._mode:
            case Tokenization.WORD:
                return " ".join(decoded_tokens)
            case Tokenization.CHAR:
                return "".join(decoded_tokens)
            case Tokenization.BPE:
                raise NotImplementedError("BPE tokenization is not implemented yet")
