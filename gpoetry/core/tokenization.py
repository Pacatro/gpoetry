from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable


class TokenizerType(Enum):
    """The type of tokenizer."""

    WORD = "word"
    CHAR = "char"


@dataclass
class TokenizerConfig:
    """Tokenizer configuration."""

    tk_type: str = TokenizerType.CHAR.value
    is_fitted: bool = False
    stoi: dict[str, int] = field(default_factory=dict)
    itos: dict[int, str] = field(default_factory=dict)
    vocab: list[str] = field(default_factory=list)
    vocab_size: int = 0


class Tokenizer(ABC):
    """Abstract base class for tokenizers."""

    def __init__(self, config: TokenizerConfig | None = None):
        """Initializes the tokenizer.

        Args:
            config (TokenizerConfig | None, optional): The tokenizer configuration. Defaults to None.
        """
        self.config = config or TokenizerConfig()

    @abstractmethod
    def _tokenize(self, text: str) -> list[str]:
        """Tokenizes a string.

        Args:
            text (str): The string to tokenize.

        Raises:
            NotImplementedError: This method is not implemented.

        Returns:
            list[str]: A list of tokens.
        """
        raise NotImplementedError("Tokenize method is not implemented")

    @abstractmethod
    def _detokenize(self, tokens: list[str]) -> str:
        """Detokenizes a list of tokens.

        Args:
            tokens (list[str]): The list of tokens to detokenize.

        Raises:
            NotImplementedError: This method is not implemented.

        Returns:
            str: The detokenized string.
        """
        raise NotImplementedError("Detokenize method is not implemented")

    def fit(self, texts: list[str] | str) -> None:
        """Fits the tokenizer to a corpus.

        Args:
            texts (list[str] | str): The corpus to fit the tokenizer to.

        Raises:
            RuntimeError: If the tokenizer is already fitted.
        """
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
        """Encodes a string into a list of integers.

        Args:
            text (str): The string to encode.

        Raises:
            RuntimeError: If the tokenizer is not fitted.

        Returns:
            list[int]: The encoded string.
        """
        if not self.config.is_fitted:
            raise RuntimeError("Tokenizer is not fitted")

        tokens = self._tokenize(text)
        return [self.config.stoi[t] for t in tokens]

    def decode(self, tokens: list[int]) -> str:
        """Decodes a list of integers into a string.

        Args:
            tokens (list[int]): The list of integers to decode.

        Raises:
            RuntimeError: If the tokenizer is not fitted.

        Returns:
            str: The decoded string.
        """
        assert len(tokens) > 0, "Cannot decode empty tokens"

        if not self.config.is_fitted:
            raise RuntimeError("Tokenizer is not fitted")

        decoded_tokens = [self.config.itos[t] for t in tokens]
        return self._detokenize(decoded_tokens)


class WordTokenizer(Tokenizer):
    """A word-based tokenizer."""

    def _tokenize(self, text: str) -> list[str]:
        """Tokenizes a string by splitting it into words.

        Args:
            text (str): The string to tokenize.

        Returns:
            list[str]: A list of words.
        """
        return text.split()

    def _detokenize(self, tokens: list[str]) -> str:
        """Detokenizes a list of words by joining them with spaces.

        Args:
            tokens (list[str]): The list of words to detokenize.

        Returns:
            str: The detokenized string.
        """
        return " ".join(tokens)


class CharTokenizer(Tokenizer):
    """A character-based tokenizer."""

    def _tokenize(self, text: str) -> list[str]:
        """Tokenizes a string by splitting it into characters.

        Args:
            text (str): The string to tokenize.

        Returns:
            list[str]: A list of characters.
        """
        return list(text)

    def _detokenize(self, tokens: list[str]) -> str:
        """Detokenizes a list of characters by joining them.

        Args:
            tokens (list[str]): The list of characters to detokenize.

        Returns:
            str: The detokenized string.
        """
        return "".join(tokens)


type TokenizerConstructor = Callable[[TokenizerConfig], Tokenizer]

tokenizers: dict[TokenizerType, TokenizerConstructor] = {
    TokenizerType.WORD: WordTokenizer,
    TokenizerType.CHAR: CharTokenizer,
}


def get_tokenizer(config: TokenizerConfig) -> Tokenizer:
    """Gets a tokenizer by type.

    Args:
        tk_type (str): The type of tokenizer.

    Returns:
        Tokenizer: The tokenizer.

    """
    return tokenizers[TokenizerType(config.tk_type)](config)
