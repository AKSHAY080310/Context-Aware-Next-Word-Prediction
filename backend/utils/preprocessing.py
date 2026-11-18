
from nltk.tokenize import word_tokenize


def tokenize_text(text):

    if not isinstance(text, str):
        raise ValueError("Input must be a string.")

    return word_tokenize(text)


def prepare_input(
    text,
    word_to_id,
    context_size=10
):

    tokens = tokenize_text(text)

    if not tokens:
        raise ValueError("Please enter some text.")

    unk_id = word_to_id.get("<UNK>", 1)
    pad_id = word_to_id.get("<PAD>", 0)

    token_ids = [
        word_to_id.get(token, unk_id)
        for token in tokens
    ][-context_size:]

    token_ids = (
        [pad_id] * (context_size - len(token_ids))
        + token_ids
    )

    return token_ids, tokens
