from transformers import T5ForConditionalGeneration, T5Tokenizer

def load_model():
    model = T5ForConditionalGeneration.from_pretrained("app/model")
    tokenizer = T5Tokenizer.from_pretrained("app/model")
    return model, tokenizer
