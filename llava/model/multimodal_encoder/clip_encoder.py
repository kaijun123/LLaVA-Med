import torch
import torch.nn as nn
from PIL import Image
from llava.mm_utils import process_images

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        print("llava-med CLIPVisionTower")
        super().__init__()
        self.vision_tower_name = vision_tower
        self.vision_tower_path = getattr(args, "vision_tower_path", "")
        self.image_processor_path = getattr(args, "image_processor_path", "")
        print("self.vision_tower_name:", self.vision_tower_name)
        print("self.vision_tower_path:", self.vision_tower_path)
        print("self.image_processor_path:", self.image_processor_path)

        self.select_layer = getattr(args, "mm_vision_select_layer", -2)
        print("self.select_layer:", self.select_layer)
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        print("self.select_feature:", self.select_feature)
        self.image_aspect_ratio = getattr(args, "image_aspect_ratio", "pad")
        self.is_loaded = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not delay_load:
            self.load_model()
        else:
            # just obtain the config of the model only
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
            # print("self.cfg_only:", self.cfg_only)

    def load_model(self):
        # print("calling CLIPVisionTower.load_model()")
        if self.is_loaded:
            print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        if self.vision_tower_path == "" or self.image_processor_path == "":
            print("self.vision_tower_name:", self.vision_tower_name)
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        else:
            print("self.vision_tower_path:", self.vision_tower_path)
            print("self.image_processor_path:", self.image_processor_path)
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_path)
            self.vision_tower = CLIPVisionModel.from_pretrained(self.image_processor_path)

        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        elif self.select_feature == 'cls':
            image_features = image_features[:, 0]
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype), output_hidden_states=True
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    # @property
    # def device(self):
    #     return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class CustomCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.vision_tower_name = vision_tower

        self.select_layer = getattr(args, "mm_vision_select_layer", -2)
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        print("self.select_feature:", self.select_feature)
        self.image_aspect_ratio = getattr(args, "image_aspect_ratio", "pad")
        self.is_loaded = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
            # print("self.cfg_only:", self.cfg_only)

    def load_model(self):
        if self.is_loaded:
            print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        print("self.vision_tower_name:", self.vision_tower_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)

        self.is_loaded = True

    def load_custom_model(self, vision_tower_path, image_processor_path):
        if self.is_loaded:
            print("{} is already loaded, `load_custom_model` called again, skipping.".format(self.vision_tower_name))
            return

        self.vision_tower_path = vision_tower_path
        self.image_processor_path = image_processor_path
        print("self.vision_tower_path:", self.vision_tower_path)
        print("self.image_processor_path:", self.image_processor_path)

        self.image_processor = CLIPImageProcessor.from_pretrained(self.image_processor_path)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        """Returns the CLS token and the patch embeddings (ie select_feature == 'cls_patch')"""
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
            # TODO: Add additional processing methods to pool the results in each of the patch embeddings
        elif self.select_feature == "cls":
            image_features = image_features[:, 0]
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    # def get_tokens(self, select_feature, image_features):
    #     """
    #     Function to obtain the CLS/ patch tokens after extracting the image features.
    #     Can only be used when "select_feature" is "cls_patch"
    #     """
    #     if select_feature == "patch":
    #         image_features = image_features[:, 1:]
    #     elif select_feature == "cls":
    #         image_features = image_features[:, 0]
    #     else:
    #         raise ValueError(f"Unexpected select feature: {self.select_feature}")

    #     return image_features

    def preprocess(self, image_paths):
        if not self.is_loaded:
            raise ValueError(f"Image processor is not loaded yet")
        images = []
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            images.append(image)

        return process_images(images, self.image_processor, self.config)

    # @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    # @property
    # def device(self):
    #     return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
