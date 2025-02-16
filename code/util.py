# import os
# import textwrap
# from io import BytesIO
from typing import List

# import requests
import torch
# import transformers
from PIL import Image
# import nibabel as nib

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    # get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

# # Normalize and convert NIfTI slices to PIL.Image
# def nifti_to_pil_slices(img_data):
#     slices = []
#     # Iterate over slices in the 3rd dimension (depth)
#     for i in range(img_data.shape[2]):
#         slice_data = img_data[:, :, i]

#         # # Normalize the slice to be in range [0, 255]
#         # slice_normalized = 255 * (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))
#         # # Convert to uint8 type for PIL compatibility
#         # slice_uint8 = slice_normalized.astype(np.uint8)

#         # Convert to uint8 type for PIL compatibility
#         slice_uint8 = slice_data.astype(np.uint8)

#         # Convert to PIL.Image (mode 'L' for grayscale)
#         pil_image = Image.fromarray(slice_uint8, mode='L')
#         slices.append(pil_image)
#     return slices


CONV_MODE = "mistral_instruct"
def create_prompt(prompt: str):
    conv = conv_templates[CONV_MODE].copy()
    roles = conv.roles
    # prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv.append_message(roles[0], prompt)
    conv.append_message(roles[1], None)
    return conv.get_prompt(), conv


def get_prediction(model, tokenizer, image_processor, image_url: str, question: str):
    image = Image.open(image_url)
    image_tensor = process_images([image], image_processor, model.config)[0]
    # print("image_tensor:", image_tensor)

    prompt, conv = create_prompt(question)
    print("prompt:", prompt)
    print("conv:", conv)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    )
    # print("model.device:", model.device)
    print("input_ids:", input_ids)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    # stopping_criteria = KeywordsStoppingCriteria(keywords=[stop_str], tokenizer=tokenizer, input_ids=input_ids)

    # taken from model_vqa and the arguments provided in the readme
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


# # Load the NIfTI file (replace with your file path)
# file_path = "../Datasets/archive/BraTS-GLI-00049-000/BraTS-GLI-00049-000-t1n.nii/00000009_brain_t1.nii"
# img = nib.load(file_path)

# # Access the image data as a NumPy array
# img_data = img.get_fdata()

# # Display some basic information
# print(f"Image shape: {img_data.shape}")
# print(f"Image affine matrix:\n{img.affine}")
# print(f"Header information:\n{img.header}")

# # download the huggging face model
# tokenizer, model, image_processor, context_len = load_pretrained_model(
#      model_path='microsoft/llava-med-v1.5-mistral-7b',
#      model_base=None,
#      model_name='llava-med-v1.5-mistral-7b',
#      device='cuda'
# )
