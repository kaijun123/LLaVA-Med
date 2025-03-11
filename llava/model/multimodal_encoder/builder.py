import os
from .clip_encoder import CLIPVisionTower, CustomCLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    # pass in mm_vision_tower/ vision_tower to specify which vision model to use
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)


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

    vision_tower_path = "/home/r11kaijun/LLaVA/llava/model/multimodal_encoder/vision_tower-epoch-1-lr-0.0001"
    image_processor_path = "/home/r11kaijun/LLaVA/llava/model/multimodal_encoder/vision_tower-epoch-1-lr-0.0001"

    vision_tower_instance = build_vision_tower(
        ModelArguments(
            vision_tower=vision_tower_path,
            mm_vision_select_feature="patch",
            vision_tower_path=vision_tower_path,
            image_processor_path=image_processor_path,
        )
    )

    # print(vision_tower_instance.vision_tower)
    # print(vision_tower_instance.image_processor)
