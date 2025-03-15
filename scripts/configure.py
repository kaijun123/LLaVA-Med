import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.model.multimodal_encoder.builder import build_vision_tower


def merge_weights(model_path, model_base, model_name, save_model_path):
    """
    Combine the base model's weights and the LoRA weights, and save the weights
    """
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, device_map='cpu'
    )

    model.save_pretrained(save_model_path)
    tokenizer.save_pretrained(save_model_path)


def configure_vision_tower(model_path, model_base, model_name, model_args, save_model_path):
    """
    Load the model from a checkpoint. Swap out the vision_tower with other weights and then save the weights
    """
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, device_map='cpu'
    )
    print("loading model")
    print("model:", model)
    # print("state dict:", model.state_dict().keys())

    print("loading vision_tower")
    custom_vision_tower = build_vision_tower(model_args)
    print("custom_vision_tower:", custom_vision_tower)

    model.model.vision_tower = custom_vision_tower
    # print("model:", model)
    # print("state dict:", model.state_dict().keys())

    # for name, param in custom_vision_tower.named_parameters():
    #     if "vision_tower" in name:
    #       print(name, param)
    #       break

    # for name, param in model.named_parameters():
    #     if "vision_tower" in name:
    #       print(name, param)
    #       break

    model.save_pretrained(save_model_path)
    tokenizer.save_pretrained(save_model_path)


if __name__ == "__main__":
    from dataclasses import dataclass, field
    from typing import Optional

    @dataclass
    class ModelArguments:
        model_name_or_path: Optional[str] = field(default="microsoft/llava-med-v1.5-mistral-7b")
        version: Optional[str] = field(default="mistral_instruct")
        freeze_backbone: bool = field(default=False)
        tune_mm_mlp_adapter: bool = field(default=False)
        vision_tower: Optional[str] = field(default="openai/clip-vit-large-patch14-336")
        mm_vision_select_layer: Optional[int] = field(default=-2)  # default to the last layer
        pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
        mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
        mm_use_im_start_end: bool = field(default=False)
        mm_use_im_patch_token: bool = field(default=True)
        mm_patch_merge_type: Optional[str] = field(default="flat")
        mm_vision_select_feature: Optional[str] = field(default="cls")
        vision_tower_path: Optional[str] = field(default="")
        image_processor_path: Optional[str] = field(default="")

    vision_tower_path = "/home/r11kaijun/LLaVA-Med/checkpoints/vision_tower-epoch-1-lr-0.0001"
    image_processor_path = "/home/r11kaijun/LLaVA-Med/checkpoints/vision_tower-epoch-1-lr-0.0001"

    configure_vision_tower(
        model_path="microsoft/llava-med-v1.5-mistral-7b",
        model_base="microsoft/llava-med-v1.5-mistral-7b",
        model_name="microsoft/llava-med-v1.5-mistral-7b",
        model_args=ModelArguments(vision_tower_path=vision_tower_path, image_processor_path=image_processor_path),
        save_model_path="/home/r11kaijun/LLaVA-Med/checkpoints/llava-med-v1.5-mistral-7b-vision_tower-epoch-1-lr-0.0001-v2",
    )
