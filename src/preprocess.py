def preprocess_texts(texts):
    """
    Lowercase and remove HTML line breaks
    """
    return [t.lower().replace("<br />", " ") for t in texts]
