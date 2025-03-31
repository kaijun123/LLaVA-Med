# from utils import validate
# from llava.model.builder import load_pretrained_model
# import argparse


if __name__ == "__main__":
    print("hello")
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, required=True)
    # parser.add_argument("--model_base", type=str, required=True)
    # parser.add_argument("--model_name", type=str, required=True)
    # parser.add_argument("--image_processor_path", type=str, required=True)
    # parser.add_argument("--vision_tower_path", type=str, required=True)
    # parser.add_argument("--image-base-path", type=str, required=True)
    # parser.add_argument("--question-file", type=str, required=True)
    # parser.add_argument("--answers-file", type=str, required=True)
    # args = parser.parse_args()

    # model_path=args.model_path
    # model_base=args.model_base if args.model_base != "" else None
    # model_name=args.model_name
    # image_processor_path=args.image_processor_path
    # vision_tower_path=args.vision_tower_path
    # question_file=args.question_file
    # answers_file=args.answers_file
    # image_base_path=args.image_base_path

    # print("llava validate.py")
    # print("model_path:", model_path)
    # print("model_base:", model_base)
    # print("model_name:", model_name)
    # print("image_processor_path:", image_processor_path)
    # print("vision_tower_path:", vision_tower_path)
    # print("image_base_path:", image_base_path)

    # finetuned_tokenizer, finetuned_model, finetuned_image_processor, finetuned_context_len = load_pretrained_model(
    #   model_path=model_path,
    #   model_base=model_base,
    #   model_name=model_name,
    #   vision_tower_path=vision_tower_path,
    #   image_processor_path=image_processor_path,
    #   device="cuda",
    # )

    # print("finetuned_model:", finetuned_model)
    # print("finetuned_context_len:", finetuned_context_len)

    # validate(
    #     data_path=question_file,
    #     image_base_path=image_base_path,
    #     model=finetuned_model,
    #     tokenizer=finetuned_tokenizer,
    #     image_processor=finetuned_image_processor,
    #     output_path=answers_file
    # )
