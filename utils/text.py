import tiktoken


def get_encoder(model: str):
    try:
        encoder = tiktoken.encoding_for_model(model)
        return encoder.encode
    except Exception as ae:
        encoder = tiktoken.get_encoding("o200k_base")
        return encoder.encode


def count_tokens(text, model):
    tokenizer = get_encoder(model)

    if tokenizer:
        return len(tokenizer(text))
