import pytest

from gpoetry.tokenization import Tokenization, Tokenizer


def test_word_tokenization():
    text = "This is a test"
    tokenizer = Tokenizer(Tokenization.WORD)
    tokenizer.fit([text])
    encoded_string = tokenizer.encode(text)
    decoded_string = tokenizer.decode(encoded_string)

    assert tokenizer.vocab_size == 4
    assert encoded_string == [0, 2, 1, 3]
    assert decoded_string == text


def test_char_tokenization():
    tokenizer = Tokenizer(Tokenization.CHAR)
    text = "This is a test"
    tokenizer.fit([text])
    encoded_string = tokenizer.encode(text)
    decoded_string = tokenizer.decode(encoded_string)

    assert tokenizer.vocab_size == 8
    assert encoded_string == [1, 4, 5, 6, 0, 5, 6, 0, 2, 0, 7, 3, 6, 7]
    assert decoded_string == text


def test_tokenizer_not_fitted():
    tokenizer = Tokenizer(Tokenization.WORD)
    text = "This is a test"

    with pytest.raises(RuntimeError) as excinfo:
        _ = tokenizer.encode(text)

    assert "Tokenizer is not fitted" in str(excinfo.value)
