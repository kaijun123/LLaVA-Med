from llava.model.builder import load_pretrained_model
from PIL import Image
import json
import os
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    # get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.conversation import SeparatorStyle, conv_templates
import torch
from llava.conversation import SeparatorStyle, conv_templates

# tokenizer, model, image_processor, context_len = load_pretrained_model(
#     model_path='microsoft/llava-med-v1.5-mistral-7b',
#     model_base=None,
#     model_name='llava-med-v1.5-mistral-7b',
#     device='cuda'
# )


def load_model(model_path, model_base, model_name):
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
        device='cuda',
    )
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device="cuda", dtype=torch.float16)
    model.model.mm_projector.to(device="cuda", dtype=torch.float16)
    model.to(device="cuda", dtype=torch.float16)
    image_processor = vision_tower.image_processor

    return tokenizer, model, image_processor, context_len


def load_base_model():
    return load_model(
        model_path="",
        model_base="microsoft/llava-med-v1.5-mistral-7b",
        model_name="microsoft/llava-med-v1.5-mistral-7b",
    )


CONV_MODE = "mistral_instruct"


def create_prompt(prompt: str):
    conv = conv_templates[CONV_MODE].copy()
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
            do_sample=False,
            temperature=0,
            top_p=None,
            num_beams=None,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


def validate(data_path, image_base_path, model, tokenizer, image_processor, output_path):
    list_data_dict = json.load(open(data_path))
    results = []
    count = 0

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
            "study_id", study_id,
            "prompt", question,
            "image", image_url,
            "ground_truth", ground_truth,
            "prediction", prediction
        )
        
        count += 1
        if count == 20:
            break

    file = open(output_path, mode="w")
    json.dump(results, file)
    file.close()


# default settings trained on full dataset, with 4 bits quantization
finetuned_tokenizer, finetuned_model, finetuned_image_processor, finetuned_context_len = load_model(
    model_path='../checkpoints/train_5k_quantized_4-epoch-3-lr-2e5',
    model_base='microsoft/llava-med-v1.5-mistral-7b',
    model_name='train_5k_quantized_4-epoch-3-lr-2e5',
)

validate(
    data_path="/home/r11kaijun/MIMIC-CXR/processed_data/validate.json",
    image_base_path="/home/r11kaijun/physionet.org/files/mimic-cxr-jpg/2.1.0",
    model=finetuned_model,
    tokenizer=finetuned_tokenizer,
    image_processor=finetuned_image_processor,
    output_path="/home/r11kaijun/MIMIC-CXR/validation_results/train_5k_quantized_4-epoch-3-lr-2e5.json",
)
