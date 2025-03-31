from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.model.multimodal_encoder.builder import build_vision_tower


def configure_vision_tower(model_path, model_base, model_name, model_args, save_model_path):
    """
    Load the model from a checkpoint. Replace the vision_tower with another vision_tower and then save the weights

    If replacing the vision tower in a pretrained model: model_base = None, model_path = path to pretrained model
    If replacing the vision tower in finetuned model, model_base = path to pretrained model, model_path = path to LoRA weights
    """

    vision_tower_path, image_processor_path = model_args.vision_tower_path, model_args.image_processor_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, vision_tower_path, image_processor_path, device_map='cpu'
    )
    print("loading model")
    print("model:", model)

    print("loading vision_tower")
    custom_vision_tower = build_vision_tower(model_args)
    print("custom_vision_tower:", custom_vision_tower)

    # replace the vision_tower in the pretrained model with our own custom finetuned vision_towere
    model.model.vision_tower = custom_vision_tower

    # additional code added just to make sure that the replacement of the vision_tower is correct
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

    vision_tower_path = "/home/FYP/angk0064/ANGK0064/checkpoints/vision_tower-epoch-1-lr-0.0001"
    image_processor_path = "/home/FYP/angk0064/ANGK0064/checkpoints/vision_tower-epoch-1-lr-0.0001"

    # microsoft/llava-med-v1.5-mistral-7b
    configure_vision_tower(
        model_path="microsoft/llava-med-v1.5-mistral-7b",
        model_base=None,
        model_name="llava-med-v1.5-mistral-7b",
        model_args=ModelArguments(vision_tower_path=vision_tower_path, image_processor_path=image_processor_path),
        save_model_path="/home/FYP/angk0064/ANGK0064/checkpoints/llava-med-v1.5-mistral-7b-vision_tower-epoch-1-lr-0.0001-v2",
    )
