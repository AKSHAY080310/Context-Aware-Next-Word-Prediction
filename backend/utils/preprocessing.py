from nltk.tokenize import word_tokenize

def tokenize_text(text):
    """
    Tokenize input using the same NLTK tokenizer
    used during model training.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")

    return word_tokenize(text)

def prepare_input(text, word_to_id, context_size=10):
    """
    Convert text into token IDs and prepare a fixed-length
    context window for the LSTM.
    """

    tokens = tokenize_text(text)

    if not tokens:
        raise ValueError("Please enter some text.")

    unk_id = word_to_id["<UNK>"]
    pad_id = word_to_id["<PAD>"]

    token_ids = [
        word_to_id.get(token, unk_id)
        for token in tokens
    ]

    # Keep only the latest 10 tokens.
    token_ids = token_ids[-context_size:]

    # Left-pad shorter inputs.
    if len(token_ids) < context_size:
        padding_length = context_size - len(token_ids)

        token_ids = (
            [pad_id] * padding_length + token_ids
        )

    return token_ids, tokens