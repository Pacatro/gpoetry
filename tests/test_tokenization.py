import pytest

from gpoetry.tokenization import WordTokenizer, CharTokenizer


class TestWordTokenizer:
    """Tests for the WordTokenizer class."""

    @pytest.fixture
    def tokenizer(self):
        """Create a fresh WordTokenizer instance for each test."""
        return WordTokenizer()

    @pytest.fixture
    def fitted_tokenizer(self):
        """Create a fitted WordTokenizer with sample data."""
        tokenizer = WordTokenizer()
        tokenizer.fit(["This is a test", "Another test sentence"])
        return tokenizer

    def test_initialization(self, tokenizer):
        """Test that WordTokenizer initializes correctly."""
        assert not tokenizer._is_fitted
        assert tokenizer._stoi == {}
        assert tokenizer._itos == {}
        assert tokenizer.vocab == []

    def test_fit_single_text(self, tokenizer):
        """Test fitting with a single text string."""
        text = "This is a test"
        tokenizer.fit(text)

        assert tokenizer._is_fitted
        assert tokenizer.vocab_size == 4
        assert set(tokenizer.vocab) == {"This", "is", "a", "test"}
        assert len(tokenizer._stoi) == 4
        assert len(tokenizer._itos) == 4

    def test_fit_list_of_texts(self, tokenizer):
        """Test fitting with a list of texts."""
        texts = ["This is a test", "Another test sentence"]
        tokenizer.fit(texts)

        assert tokenizer._is_fitted
        assert tokenizer.vocab_size == 6
        assert set(tokenizer.vocab) == {
            "This",
            "is",
            "a",
            "test",
            "Another",
            "sentence",
        }

    def test_fit_vocab_sorted(self, tokenizer):
        """Test that vocabulary is sorted alphabetically."""
        tokenizer.fit("zebra apple banana")

        assert tokenizer.vocab == ["apple", "banana", "zebra"]
        assert tokenizer._stoi["apple"] == 0
        assert tokenizer._stoi["banana"] == 1
        assert tokenizer._stoi["zebra"] == 2

    def test_fit_already_fitted_raises_error(self, fitted_tokenizer):
        """Test that fitting an already fitted tokenizer raises an error."""
        with pytest.raises(RuntimeError) as excinfo:
            fitted_tokenizer.fit("New text")

        assert "already fitted" in str(excinfo.value)

    def test_fit_with_empty_strings(self, tokenizer):
        """Test fitting with empty strings in the list."""
        texts = ["This is", "", "a test", ""]
        tokenizer.fit(texts)

        assert tokenizer.vocab_size == 4
        assert set(tokenizer.vocab) == {"This", "is", "a", "test"}

    def test_encode_basic(self, fitted_tokenizer):
        """Test basic encoding of text."""
        # fitted_tokenizer has vocab: ["Another", "This", "a", "is", "sentence", "test"]
        encoded = fitted_tokenizer.encode("This is a test")

        assert isinstance(encoded, list)
        assert all(isinstance(token, int) for token in encoded)
        assert len(encoded) == 4

    def test_encode_not_fitted_raises_error(self, tokenizer):
        """Test that encoding without fitting raises an error."""
        with pytest.raises(RuntimeError) as excinfo:
            tokenizer.encode("This is a test")

        assert "not fitted" in str(excinfo.value)

    def test_encode_decode_roundtrip(self, tokenizer):
        """Test that encoding and decoding produces the original text."""
        text = "This is a test"
        tokenizer.fit(text)

        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        assert decoded == text

    def test_encode_decode_multiple_texts(self, fitted_tokenizer):
        """Test encoding and decoding multiple different texts."""
        test_cases = [
            "This is a test",
            "Another test",
            "This sentence",
        ]

        for text in test_cases:
            encoded = fitted_tokenizer.encode(text)
            decoded = fitted_tokenizer.decode(encoded)
            assert decoded == text

    def test_decode_not_fitted_raises_error(self, tokenizer):
        """Test that decoding without fitting raises an error."""
        with pytest.raises(RuntimeError) as excinfo:
            tokenizer.decode([0, 1, 2])

        assert "not fitted" in str(excinfo.value)

    def test_decode_empty_tokens_raises_error(self, fitted_tokenizer):
        """Test that decoding empty tokens raises an assertion error."""
        with pytest.raises(AssertionError) as excinfo:
            fitted_tokenizer.decode([])

        assert "Cannot decode empty tokens" in str(excinfo.value)

    def test_encode_preserves_word_order(self, tokenizer):
        """Test that encoding preserves word order."""
        tokenizer.fit("word1 word2 word3")

        encoded1 = tokenizer.encode("word1 word2")
        encoded2 = tokenizer.encode("word2 word1")

        assert encoded1 != encoded2
        assert encoded1[0] == encoded2[1]
        assert encoded1[1] == encoded2[0]

    def test_stoi_itos_consistency(self, fitted_tokenizer):
        """Test that stoi and itos mappings are consistent."""
        for token, idx in fitted_tokenizer._stoi.items():
            assert fitted_tokenizer._itos[idx] == token

        for idx, token in fitted_tokenizer._itos.items():
            assert fitted_tokenizer._stoi[token] == idx

    def test_repeated_words(self, tokenizer):
        """Test handling of repeated words in text."""
        text = "test test test"
        tokenizer.fit(text)

        assert tokenizer.vocab_size == 1
        assert tokenizer.vocab == ["test"]

        encoded = tokenizer.encode(text)
        assert encoded == [0, 0, 0]
        assert tokenizer.decode(encoded) == text

    def test_single_word(self, tokenizer):
        """Test tokenization of a single word."""
        tokenizer.fit("hello")

        assert tokenizer.vocab_size == 1
        encoded = tokenizer.encode("hello")
        assert encoded == [0]
        assert tokenizer.decode(encoded) == "hello"

    @pytest.mark.parametrize(
        "text,expected_vocab_size",
        [
            ("one two three", 3),
            ("a b c d e", 5),
            ("test", 1),
            ("same same same", 1),
            ("hello world hello", 2),
        ],
    )
    def test_various_texts(self, tokenizer, text, expected_vocab_size):
        """Test tokenization with various text patterns."""
        tokenizer.fit(text)
        assert tokenizer.vocab_size == expected_vocab_size

        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text


class TestCharTokenizer:
    """Tests for the CharTokenizer class."""

    @pytest.fixture
    def tokenizer(self):
        """Create a fresh CharTokenizer instance for each test."""
        return CharTokenizer()

    @pytest.fixture
    def fitted_tokenizer(self):
        """Create a fitted CharTokenizer with sample data."""
        tokenizer = CharTokenizer()
        tokenizer.fit(["This is a test", "Another test!"])
        return tokenizer

    def test_initialization(self, tokenizer):
        """Test that CharTokenizer initializes correctly."""
        assert not tokenizer._is_fitted
        assert tokenizer._stoi == {}
        assert tokenizer._itos == {}
        assert tokenizer.vocab == []

    def test_fit_single_text(self, tokenizer):
        """Test fitting with a single text string."""
        text = "This is a test"
        tokenizer.fit(text)

        assert tokenizer._is_fitted
        assert tokenizer.vocab_size == 8  # 'T', 'h', 'i', 's', ' ', 'a', 't', 'e'
        unique_chars = set(text)
        assert set(tokenizer.vocab) == unique_chars

    def test_fit_list_of_texts(self, tokenizer):
        """Test fitting with a list of texts."""
        texts = ["abc", "def"]
        tokenizer.fit(texts)

        assert tokenizer._is_fitted
        assert tokenizer.vocab_size == 6
        assert set(tokenizer.vocab) == {"a", "b", "c", "d", "e", "f"}

    def test_fit_vocab_sorted(self, tokenizer):
        """Test that vocabulary is sorted alphabetically."""
        tokenizer.fit("zxy")

        assert tokenizer.vocab == ["x", "y", "z"]
        assert tokenizer._stoi["x"] == 0
        assert tokenizer._stoi["y"] == 1
        assert tokenizer._stoi["z"] == 2

    def test_fit_already_fitted_raises_error(self, fitted_tokenizer):
        """Test that fitting an already fitted tokenizer raises an error."""
        with pytest.raises(RuntimeError) as excinfo:
            fitted_tokenizer.fit("New text")

        assert "already fitted" in str(excinfo.value)

    def test_fit_with_empty_strings(self, tokenizer):
        """Test fitting with empty strings in the list."""
        texts = ["abc", "", "def", ""]
        tokenizer.fit(texts)

        assert tokenizer.vocab_size == 6
        assert set(tokenizer.vocab) == {"a", "b", "c", "d", "e", "f"}

    def test_encode_basic(self, fitted_tokenizer):
        """Test basic encoding of text."""
        encoded = fitted_tokenizer.encode("This")

        assert isinstance(encoded, list)
        assert all(isinstance(token, int) for token in encoded)
        assert len(encoded) == 4

    def test_encode_not_fitted_raises_error(self, tokenizer):
        """Test that encoding without fitting raises an error."""
        with pytest.raises(RuntimeError) as excinfo:
            tokenizer.encode("text")

        assert "not fitted" in str(excinfo.value)

    def test_encode_decode_roundtrip(self, tokenizer):
        """Test that encoding and decoding produces the original text."""
        text = "Hello, World!"
        tokenizer.fit(text)

        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        assert decoded == text

    def test_encode_decode_multiple_texts(self, fitted_tokenizer):
        """Test encoding and decoding multiple different texts."""
        test_cases = [
            "This is",
            "test!",
            "Another",
        ]

        for text in test_cases:
            encoded = fitted_tokenizer.encode(text)
            decoded = fitted_tokenizer.decode(encoded)
            assert decoded == text

    def test_decode_not_fitted_raises_error(self, tokenizer):
        """Test that decoding without fitting raises an error."""
        with pytest.raises(RuntimeError) as excinfo:
            tokenizer.decode([0, 1, 2])

        assert "not fitted" in str(excinfo.value)

    def test_decode_empty_tokens_raises_error(self, fitted_tokenizer):
        """Test that decoding empty tokens raises an assertion error."""
        with pytest.raises(AssertionError) as excinfo:
            fitted_tokenizer.decode([])

        assert "Cannot decode empty tokens" in str(excinfo.value)

    def test_encode_preserves_char_order(self, tokenizer):
        """Test that encoding preserves character order."""
        tokenizer.fit("abc")

        encoded_abc = tokenizer.encode("abc")
        encoded_cba = tokenizer.encode("cba")

        assert encoded_abc != encoded_cba
        assert encoded_abc == list(reversed(encoded_cba))

    def test_stoi_itos_consistency(self, fitted_tokenizer):
        """Test that stoi and itos mappings are consistent."""
        for char, idx in fitted_tokenizer._stoi.items():
            assert fitted_tokenizer._itos[idx] == char

        for idx, char in fitted_tokenizer._itos.items():
            assert fitted_tokenizer._stoi[char] == idx

    def test_repeated_characters(self, tokenizer):
        """Test handling of repeated characters in text."""
        text = "aaa"
        tokenizer.fit(text)

        assert tokenizer.vocab_size == 1
        assert tokenizer.vocab == ["a"]

        encoded = tokenizer.encode(text)
        assert encoded == [0, 0, 0]
        assert tokenizer.decode(encoded) == text

    def test_single_character(self, tokenizer):
        """Test tokenization of a single character."""
        tokenizer.fit("x")

        assert tokenizer.vocab_size == 1
        encoded = tokenizer.encode("x")
        assert encoded == [0]
        assert tokenizer.decode(encoded) == "x"

    def test_special_characters(self, tokenizer):
        """Test tokenization with special characters."""
        text = "Hello, World! 123"
        tokenizer.fit(text)

        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        assert decoded == text
        assert "," in tokenizer.vocab
        assert "!" in tokenizer.vocab
        assert " " in tokenizer.vocab

    def test_whitespace_handling(self, tokenizer):
        """Test that whitespace is properly tokenized."""
        text = "a b  c"  # Multiple spaces
        tokenizer.fit(text)

        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        assert decoded == text
        assert " " in tokenizer.vocab

    @pytest.mark.parametrize(
        "text,expected_vocab_size",
        [
            ("abc", 3),
            ("aaa", 1),
            ("hello", 4),  # h, e, l, o
            ("Hi!", 3),  # H, i, !
            ("123", 3),  # 1, 2, 3
        ],
    )
    def test_various_texts(self, tokenizer, text, expected_vocab_size):
        """Test tokenization with various text patterns."""
        tokenizer.fit(text)
        assert tokenizer.vocab_size == expected_vocab_size

        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
