from llava.model.builder import load_pretrained_model
from PIL import Image
import json
import os
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import process_images, tokenizer_image_token
from llava.conversation import conv_templates
import torch


def load_model(model_path, model_base, model_name, vision_tower_path, image_processor_path, device="cuda"):
    """
    Used to load weights for the model. There are 2 uses for this function
    1) Load pretrained weights: 
    Set model_base=None, model_path=pretrained VLM checkpoint, vision_tower_path=pretrained vision tower checkpoint, image_processor_path=pretrained image processor

    2) Load finetuned model using lora weights:
    Set model_base=pretrained VLM checkpoint, model_path=lora weights, vision_tower_path= vision tower checkpoint, image_processor_path=pretrained image processor
    Ensure that model_base is not None so that the lora weights will be merged when loading the model weights

    vision_tower_path and image_processor_path are provided to instantiate the VLM with a custom vision encoder. It depends on local changes made to the LlavaMistralForCausalLM code.
    Instantiating the model using other classes eg AutoModelForCausalLM will not have this effect.
    Ensure that "llava" is in "model_base" to instantiate the model using the local LlavaMistralForCausalLM class.
    """
    print("llava load_model")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
        vision_tower_path=vision_tower_path,
        image_processor_path=image_processor_path,
        device=device,
    )

    return tokenizer, model, image_processor, context_len


def load_llava_med_base_model():
    """
    Load the microsoft/llava-med-v1.5-mistral-7b model, with the pretrained vision tower.
    If a custom vision tower is to be used, replace the vision_tower_path and image_processor_path
    """
    return load_model(
        model_path="microsoft/llava-med-v1.5-mistral-7b",
        model_base=None,
        model_name="llava-med-v1.5-mistral-7b",
        vision_tower_path="openai/clip-vit-large-patch14-336",
        image_processor_path="openai/clip-vit-large-patch14-336"
    )


def load_llava_base_model():
    """
    Load the liuhaotian/llava-v1.5-7b model, with the pretrained vision tower.
    If a custom vision tower is to be used, replace the vision_tower_path and image_processor_path
    """
    return load_model(
        model_path="liuhaotian/llava-v1.5-7b",
        model_base=None,
        model_name="llava-v1.5-7b",
        vision_tower_path="openai/clip-vit-large-patch14-336",
        image_processor_path="openai/clip-vit-large-patch14-336"
    )

# mistral_instruct is used in this work, due to the mistral llm used and with deliberate reference to the readme.
# However, the system message has been edited for this work. Refer to llava/conversation.py
CONV_MODE = "mistral_instruct"

def create_prompt(prompt: str):
    """
    Create the prompt for the specified CONV_MODE.
    Edit the CONV_MODE to use other conversational templates
    """
    conv = conv_templates[CONV_MODE].copy()
    roles = conv.roles
    conv.append_message(roles[0], prompt)
    conv.append_message(roles[1], None)
    return conv.get_prompt(), conv


def get_prediction(model, tokenizer, image_processor, image_path: str, question: str):
    """
    Generate an output for an image-question pair using the model and tokenizer.
    Configs for text generation is obtained from llava/eval/model_vqa and the default arguments provided
    """

    # open the image and process it
    image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0]

    question = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
    if model.config.mm_use_im_start_end:
        question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
    else:
        question = DEFAULT_IMAGE_TOKEN + '\n' + question

    prompt, conv = create_prompt(question)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

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


def validate(data_path, image_base_path, model, tokenizer, image_processor, output_path, limited=False):
    """
    Reads the json provided by data_path. Generates the output using the model. 
    And writes the ground truth and the generated output to the json file provided by output_path.

    image_base_path: base path to be joined with the image path provided in the data_path json file
    limited: if limited == True, only 20 ouputs will be generated; used if the validation/ test set 
    is very big, and you just want to have a glimpse of thee model's output
    """
    print("Validating the model....")
    print("Writing output to:", output_path)
    list_data_dict = json.load(open(data_path))
    results = []
    count = 0

    for data in list_data_dict:
        study_id = data["id"]
        image_path = os.path.join(image_base_path, data["image"])
        print("image_path:", image_path)
        conv = data["conversations"]
        question = conv[0]["value"]
        ground_truth = conv[1]["value"]
        prediction = get_prediction(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            image_path=image_path,
            question=question,
        )

        results.append(
            {
                "study_id": study_id,
                "prompt": question,
                "image": image_path,
                "ground_truth": ground_truth,
                "prediction": prediction,
            }
        )
        print(
            "study_id", study_id, "\n"
            "prompt", question, "\n"
            "image", image_path, "\n"
            "ground_truth", ground_truth, "\n"
            "prediction", prediction, "\n"
        )
        
        count += 1
        if limited and count == 20:
            break

    # writes to output file
    file = open(output_path, mode="w")
    json.dump(results, file)
    file.close()
