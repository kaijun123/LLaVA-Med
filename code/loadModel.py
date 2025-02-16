from llava.model.builder import load_pretrained_model

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path='microsoft/llava-med-v1.5-mistral-7b',
    model_base=None,
    model_name='llava-med-v1.5-mistral-7b',
    device='cuda'
)