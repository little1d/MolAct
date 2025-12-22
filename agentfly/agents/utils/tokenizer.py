from transformers import AutoProcessor, AutoTokenizer

def create_tokenizer(model_name_or_path: str, **kwargs):
    """Create a tokenizer with optional kwargs.
    
    Args:
        model_name_or_path: Path to the model
        **kwargs: Additional arguments to pass to AutoTokenizer.from_pretrained
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
    # Can not find the tokenizer in local directory or huggingface hub
    except OSError:
        tokenizer = None
    
    return tokenizer


def create_processor(model_name_or_path: str):
    try:
        processor = AutoProcessor.from_pretrained(model_name_or_path)
    except OSError:
        processor = None
    
    return processor