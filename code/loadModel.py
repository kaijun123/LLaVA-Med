from llava.model.builder import load_pretrained_model
from PIL import Image
import json
import os
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
)
from llava.conversation import SeparatorStyle, conv_templates
import torch


def load_model(model_path, model_base, model_name, vision_tower_path, image_processor_path, device="cuda"):
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
        vision_tower_path=vision_tower_path,
        image_processor_path=image_processor_path,
        device=device,
    )

    return tokenizer, model, image_processor, context_len


def load_base_model():
    # tokenizer, model, image_processor, context_len = load_pretrained_model(
    #     model_path='microsoft/llava-med-v1.5-mistral-7b',
    #     model_base=None,
    #     model_name='llava-med-v1.5-mistral-7b',
    #     device='cuda'
    # )
    return load_model(
        model_path="microsoft/llava-med-v1.5-mistral-7b",
        model_base=None,
        model_name="llava-med-v1.5-mistral-7b",
        vision_tower_path="openai/clip-vit-large-patch14-336",
        image_processor_path="openai/clip-vit-large-patch14-336"
    )


CONV_MODE = "mistral_instruct"


def create_prompt(prompt: str):
    conv = conv_templates[CONV_MODE].copy()
    # print("init conv:", conv)
    roles = conv.roles
    # prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv.append_message(roles[0], prompt)
    conv.append_message(roles[1], None)
    return conv.get_prompt(), conv


def get_prediction(model, tokenizer, image_processor, image_url: str, question: str):
    # print("image_url:", image_url)
    image = Image.open(image_url)
    image_tensor = process_images([image], image_processor, model.config)[0]
    # print("image_tensor:", image_tensor)

    question = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
    if model.config.mm_use_im_start_end:
        question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
    else:
        question = DEFAULT_IMAGE_TOKEN + '\n' + question

    prompt, conv = create_prompt(question)
    # print("prompt:", prompt)
    # print("conv:", conv)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    )
    # print("model.device:", model.device)
    # print("input_ids:", input_ids)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    # stopping_criteria = KeywordsStoppingCriteria(keywords=[stop_str], tokenizer=tokenizer, input_ids=input_ids)

    # taken from LLava-Med/llava/eval/model_vqa and the arguments provided in the readme
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=True,
            temperature=0.2,
            top_p=None,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=2048,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


def validate(data_path, image_base_path, model, tokenizer, image_processor, output_path):
    print("output_path:", output_path)
    list_data_dict = json.load(open(data_path))
    results = []
    # count = 0

    for data in list_data_dict:
        study_id = data["id"]
        image_url = os.path.join(image_base_path, data["image"])
        print("image_url:", image_url)
        conv = data["conversations"]
        question = conv[0]["value"]
        ground_truth = conv[1]["value"]
        prediction = get_prediction(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            image_url=image_url,
            question=question,
        )

        results.append(
            {
                "study_id": study_id,
                "prompt": question,
                "image": image_url,
                "ground_truth": ground_truth,
                "prediction": prediction,
            }
        )
        print(
            "study_id", study_id, "\n"
            "prompt", question, "\n"
            "image", image_url, "\n"
            "ground_truth", ground_truth, "\n"
            "prediction", prediction, "\n"
        )
        
        # count += 1
        # if count == 20:
        #     break

    file = open(output_path, mode="w")
    json.dump(results, file)
    file.close()


if __name__ == "__main__":
    # from transformers import AutoTokenizer

    model_path="/home/FYP/angk0064/ANGK0064/checkpoints/llava-med-v1.5-mistral-7b-vision_tower-epoch-1-lr-0.0001-v2-train_28k_custom-train-mlp-and-llm-unquantized-epoch-1-lr-6e-5"
    model_base=None
    model_name="lora-llava-med-v1.5-mistral-7b-vision_tower-epoch-1-lr-0.0001-v2-train_28k_custom-train-mlp-and-llm-unquantized-epoch-1-lr-6e-5"
    vision_tower_path="/home/FYP/angk0064/ANGK0064/checkpoints/vision_tower-epoch-1-lr-0.0001"
    # "openai/clip-vit-large-patch14-336"
    image_processor_path="/home/FYP/angk0064/ANGK0064/checkpoints/vision_tower-epoch-1-lr-0.0001"
    # "openai/clip-vit-large-patch14-336"


    # tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    # print("tokenizer:", tokenizer)

    print("model_path:", model_path)
    print("model_base:", model_base)
    print("model_name:", model_name)
    print("vision_tower_path:", vision_tower_path)
    print("image_processor_path:", image_processor_path)

    finetuned_tokenizer, finetuned_model, finetuned_image_processor, finetuned_context_len = load_model(
          model_path=model_path,
          model_base=model_base,
          model_name=model_name,
          vision_tower_path=vision_tower_path,
          image_processor_path=image_processor_path
    )

    print("finetuned_model:", finetuned_model)
    print("finetuned_context_len:", finetuned_context_len, "\n")
    validate(
        data_path="/home/FYP/angk0064/Datasets/mimic-cxr/processed_data/test_custom.json",
        image_base_path="/home/FYP/angk0064/Datasets/mimic-cxr-jpg/2.1.0",
        model=finetuned_model,
        tokenizer=finetuned_tokenizer,
        image_processor=finetuned_image_processor,
        output_path="/home/FYP/angk0064/LLaVA-Med/test-results/temp-0.2/lora-llava-med-v1.5-mistral-7b-vision_tower-epoch-1-lr-0.0001-v2-train_28k_custom-train-mlp-and-llm-unquantized-epoch-1-lr-6e-5.json"
    )
