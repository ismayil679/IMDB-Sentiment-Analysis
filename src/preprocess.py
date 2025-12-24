"""Text preprocessing utilities for IMDB sentiment analysis."""


def preprocess_texts(texts):
    """
    Apply minimal preprocessing to review texts.
    
    Preprocessing steps:
    1. Convert to lowercase for case-insensitive matching
    2. Replace HTML line breaks (<br />) with spaces
    
    Note: Intentionally minimal - TF-IDF handles most text normalization.
    No lemmatization, stemming, or stopword removal to preserve sentiment signals.
    
    Args:
        texts: List of raw review strings
    
    Returns:
        List of preprocessed review strings
    
    Example:
        >>> texts = ["Great Movie!<br />Amazing", "Terrible FILM"]
        >>> preprocessed = preprocess_texts(texts)
        >>> print(preprocessed)
        ['great movie! amazing', 'terrible film']
    """
    return [t.lower().replace("<br />", " ") for t in texts]
