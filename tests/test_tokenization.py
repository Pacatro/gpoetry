import pytest

from gpoetry.core.tokenization import (
    TokenizerConfig,
    TokenizerType,
    WordTokenizer,
    CharTokenizer,
    get_tokenizer,
)


def test_get_tokenizer():
    char_config = TokenizerConfig(tk_type="char")
    word_config = TokenizerConfig(tk_type="word")

    assert isinstance(get_tokenizer(char_config), CharTokenizer)
    assert isinstance(get_tokenizer(word_config), WordTokenizer)


def test_get_tokenizer_with_invalid_type():
    with pytest.raises(ValueError, match="'foo' is not a valid TokenizerType"):
        get_tokenizer(TokenizerConfig(tk_type="foo"))


class TestTokenizerConfig:
    """Tests for TokenizerConfig"""

    def test_default_config(self):
        config = TokenizerConfig()
        assert config.tk_type == TokenizerType.CHAR.value
        assert config.is_fitted is False
        assert config.stoi == {}
        assert config.itos == {}
        assert config.vocab == []
        assert config.vocab_size == 0

    def test_custom_config(self):
        config = TokenizerConfig(
            tk_type=TokenizerType.WORD.value,
            is_fitted=True,
            stoi={"hello": 0},
            itos={0: "hello"},
            vocab=["hello"],
            vocab_size=1,
        )
        assert config.tk_type == TokenizerType.WORD.value
        assert config.is_fitted is True
        assert config.stoi == {"hello": 0}
        assert config.itos == {0: "hello"}
        assert config.vocab == ["hello"]
        assert config.vocab_size == 1


class TestCharTokenizer:
    """Tests for CharTokenizer"""

    def test_tokenize(self):
        tokenizer = CharTokenizer()
        tokens = tokenizer._tokenize("hello")
        assert tokens == ["h", "e", "l", "l", "o"]

    def test_detokenize(self):
        tokenizer = CharTokenizer()
        text = tokenizer._detokenize(["h", "e", "l", "l", "o"])
        assert text == "hello"

    def test_fit_single_text(self):
        tokenizer = CharTokenizer()
        tokenizer.fit("hello")
        assert tokenizer.config.is_fitted is True
        assert tokenizer.config.vocab == ["e", "h", "l", "o"]
        assert tokenizer.config.vocab_size == 4
        assert tokenizer.config.stoi == {"e": 0, "h": 1, "l": 2, "o": 3}
        assert tokenizer.config.itos == {0: "e", 1: "h", 2: "l", 3: "o"}

    def test_fit_multiple_texts(self):
        tokenizer = CharTokenizer()
        tokenizer.fit(["hello", "world"])
        assert tokenizer.config.is_fitted is True
        assert tokenizer.config.vocab == ["d", "e", "h", "l", "o", "r", "w"]
        assert tokenizer.config.vocab_size == 7

    def test_fit_with_empty_strings(self):
        tokenizer = CharTokenizer()
        tokenizer.fit(["hello", "", "world"])
        assert tokenizer.config.vocab == ["d", "e", "h", "l", "o", "r", "w"]

    def test_encode(self):
        tokenizer = CharTokenizer()
        tokenizer.fit("hello")
        encoded = tokenizer.encode("hello")
        assert encoded == [1, 0, 2, 2, 3]

    def test_decode(self):
        tokenizer = CharTokenizer()
        tokenizer.fit("hello")
        decoded = tokenizer.decode([1, 0, 2, 2, 3])
        assert decoded == "hello"

    def test_encode_before_fit_raises_error(self):
        tokenizer = CharTokenizer()
        with pytest.raises(RuntimeError, match="Tokenizer is not fitted"):
            tokenizer.encode("hello")

    def test_decode_before_fit_raises_error(self):
        tokenizer = CharTokenizer()
        with pytest.raises(RuntimeError, match="Tokenizer is not fitted"):
            tokenizer.decode([0, 1, 2])

    def test_fit_twice_raises_error(self):
        tokenizer = CharTokenizer()
        tokenizer.fit("hello")
        with pytest.raises(RuntimeError, match="Tokenizer is already fitted"):
            tokenizer.fit("world")

    def test_decode_empty_tokens_raises_error(self):
        tokenizer = CharTokenizer()
        tokenizer.fit("hello")
        with pytest.raises(AssertionError, match="Cannot decode empty tokens"):
            tokenizer.decode([])

    def test_roundtrip(self):
        tokenizer = CharTokenizer()
        original_text = "hello world"
        tokenizer.fit(original_text)
        encoded = tokenizer.encode(original_text)
        decoded = tokenizer.decode(encoded)
        assert decoded == original_text


class TestWordTokenizer:
    """Tests for WordTokenizer"""

    def test_tokenize(self):
        tokenizer = WordTokenizer()
        tokens = tokenizer._tokenize("hello world")
        assert tokens == ["hello", "world"]

    def test_detokenize(self):
        tokenizer = WordTokenizer()
        text = tokenizer._detokenize(["hello", "world"])
        assert text == "hello world"

    def test_fit_single_text(self):
        tokenizer = WordTokenizer()
        tokenizer.fit("hello world")
        assert tokenizer.config.is_fitted is True
        assert tokenizer.config.vocab == ["hello", "world"]
        assert tokenizer.config.vocab_size == 2
        assert tokenizer.config.stoi == {"hello": 0, "world": 1}
        assert tokenizer.config.itos == {0: "hello", 1: "world"}

    def test_fit_multiple_texts(self):
        tokenizer = WordTokenizer()
        tokenizer.fit(["hello world", "foo bar"])
        assert tokenizer.config.is_fitted is True
        assert tokenizer.config.vocab == ["bar", "foo", "hello", "world"]
        assert tokenizer.config.vocab_size == 4

    def test_encode(self):
        tokenizer = WordTokenizer()
        tokenizer.fit("hello world")
        encoded = tokenizer.encode("hello world")
        assert encoded == [0, 1]

    def test_decode(self):
        tokenizer = WordTokenizer()
        tokenizer.fit("hello world")
        decoded = tokenizer.decode([0, 1])
        assert decoded == "hello world"

    def test_encode_before_fit_raises_error(self):
        tokenizer = WordTokenizer()
        with pytest.raises(RuntimeError, match="Tokenizer is not fitted"):
            tokenizer.encode("hello")

    def test_decode_before_fit_raises_error(self):
        tokenizer = WordTokenizer()
        with pytest.raises(RuntimeError, match="Tokenizer is not fitted"):
            tokenizer.decode([0, 1])

    def test_fit_twice_raises_error(self):
        tokenizer = WordTokenizer()
        tokenizer.fit("hello world")
        with pytest.raises(RuntimeError, match="Tokenizer is already fitted"):
            tokenizer.fit("foo bar")

    def test_decode_empty_tokens_raises_error(self):
        tokenizer = WordTokenizer()
        tokenizer.fit("hello world")
        with pytest.raises(AssertionError, match="Cannot decode empty tokens"):
            tokenizer.decode([])

    def test_roundtrip(self):
        tokenizer = WordTokenizer()
        original_text = "hello world foo bar"
        tokenizer.fit(original_text)
        encoded = tokenizer.encode(original_text)
        decoded = tokenizer.decode(encoded)
        assert decoded == original_text

    def test_multiple_spaces(self):
        tokenizer = WordTokenizer()
        text = "hello  world"  # Two spaces
        tokenizer.fit(text)
        # split() removes multiple spaces
        assert tokenizer.config.vocab == ["hello", "world"]


class TestTokenizerWithConfig:
    """Tests for Tokenizer with custom configuration"""

    def test_char_tokenizer_with_config(self):
        config = TokenizerConfig(tk_type=TokenizerType.CHAR.value)
        tokenizer = CharTokenizer(config)
        assert tokenizer.config.tk_type == TokenizerType.CHAR.value
        assert tokenizer.config.is_fitted is False

    def test_word_tokenizer_with_config(self):
        config = TokenizerConfig(tk_type=TokenizerType.WORD.value)
        tokenizer = WordTokenizer(config)
        assert tokenizer.config.tk_type == TokenizerType.WORD.value
        assert tokenizer.config.is_fitted is False

    def test_tokenizer_with_prefitted_config(self):
        config = TokenizerConfig(
            tk_type=TokenizerType.CHAR.value,
            is_fitted=True,
            stoi={"h": 0, "e": 1, "l": 2, "o": 3},
            itos={0: "h", 1: "e", 2: "l", 3: "o"},
            vocab=["h", "e", "l", "o"],
            vocab_size=4,
        )
        tokenizer = CharTokenizer(config)
        # Should be able to encode/decode without calling fit
        encoded = tokenizer.encode("hello")
        assert encoded == [0, 1, 2, 2, 3]
        decoded = tokenizer.decode(encoded)
        assert decoded == "hello"
        # Should not be able to fit again
        with pytest.raises(RuntimeError, match="Tokenizer is already fitted"):
            tokenizer.fit("world")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
