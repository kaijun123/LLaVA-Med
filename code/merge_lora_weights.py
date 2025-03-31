import argparse
from llava.model.builder import load_pretrained_model


def merge_lora(args):
    """
    Used to merge the lora weights with the pretrained weights. And saves the weights at save_model_path
    """
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, 
        args.model_base, 
        args.model_name, 
        args.vision_tower_path, 
        args.image_processor_path, 
        device_map='cpu'
    )

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)
    parser.add_argument("--vision_tower_path", type=str, required=True)
    parser.add_argument("--image_processor_path", type=str, required=True)


    args = parser.parse_args()

    merge_lora(args)