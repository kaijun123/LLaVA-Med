import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import os
import time

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from llava.mm_utils import process_images
from PIL import Image


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.df = pd.read_csv(annotations_file)
        self.img_dir = img_dir

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        """Returns the absolute image file path and list of disease classifications"""
        image = os.path.join(self.img_dir, self.df.at[idx, "image_path"])
        labels = self.df.iloc[idx, 5:].values

        labels_str = ""
        for label in labels:
            labels_str += str(label)
            labels_str += ","

        labels_str = labels_str[:-1]

        return image, labels_str


def convert_label_str(label_str_list):
    """
    label_str_list: list of strings. Each string corresponds to each image
    output: list of list of numbers. Each "sublist" contains the One-Hot encoded labels for each image
    """
    label_list_list = []

    for label_str in label_str_list:
        # print("label_str:", label_str)
        label_list = []
        for label in label_str.split(","):
            # print("label:", label)
            if label == "-1.0":
                label_list.extend([1, 0, 0, 0])
            elif label == "0.0":
                label_list.extend([0, 1, 0, 0])
            elif label == "1.0":
                label_list.extend([0, 0, 1, 0])
            elif label == "2.0":
                label_list.extend([0, 0, 0, 1])
            else:
                raise ValueError(f"Invalid label: {label}")

        label_list_list.append(label_list)

    return label_list_list


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
            print("self.cfg_only:", self.cfg_only)
        self.is_loaded = True

    def load_model(self):
        if self.is_loaded:
            print(
                "{} is already loaded, `load_model` called again, skipping.".format(
                    self.vision_tower_name
                )
            )
            return

        print("self.vision_tower_name:", self.vision_tower_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name
        )
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


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )
    print("vision_tower:", vision_tower)
    is_absolute_path_exists = os.path.exists(vision_tower)
    if (
        is_absolute_path_exists
        or vision_tower.startswith("openai")
        or vision_tower.startswith("laion")
    ):
        return CustomCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)


class CLIPDiseaseClassifier(nn.Module):
    def __init__(self, input_neurons=1024, hidden_dim=1024, output_neurons=56):
        super().__init__()
        # MLP Classification Head
        self.mlp = nn.Sequential(
            nn.Linear(input_neurons, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_neurons),  # Output: (B, 56)
        )

    def forward(self, image_features):
        output = self.mlp(image_features)  # (B, 56)
        # return output.view(-1, 14, 4)  # Reshape to (batch, 14 diseases, 4 states)
        return output


def train_clip_classifier(
    vision_tower_instance: CustomCLIPVisionTower,
    classifier: CLIPDiseaseClassifier,
    train_loader,
    val_loader,
    output_dir,
    unfreeze_layers=4,
    epochs=2,
    lr=1e-4,
):
    # send the models to the gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_tower_instance.to(device)
    classifier.to(device)

    # Freeze all layers
    for param in vision_tower_instance.parameters():
        param.requires_grad = False

    # unfreeze the layers that we want to finetune in the clip encoder
    for param in vision_tower_instance.vision_tower.vision_model.encoder.layers[
        -unfreeze_layers:
    ].parameters():
        param.requires_grad = True  # Unfreeze last 4 layers

    # Our task is a multi-task one (multi-classification, multi-label)
    # Reason: For each category, there can only be one selected class.
    # But it is multi-label since the image can show more than one category
    # But by using one-hot encoding, we can convert the multi-task learniing to a multi-label classification
    criterion = nn.BCEWithLogitsLoss() # Multi-label classification loss
    optimizer = optim.AdamW(
        list(vision_tower_instance.parameters()) + list(classifier.parameters()),
        lr=lr,
    )

    avg_train_loss_arr = []
    avg_val_loss_arr = []

    for epoch in range(epochs):
        print("epoch:", epoch)
        start_time = time.time()
        vision_tower_instance.train()
        classifier.train()
        total_loss = 0

        for batch, (image_paths, labels_str) in enumerate(train_loader):
            print("batch:", batch)
            optimizer.zero_grad()

            ground_truths = torch.Tensor(convert_label_str(labels_str)).to(device)
            # print("ground_truths:", ground_truths)

            # preprocess images
            images = vision_tower_instance.preprocess(image_paths).to(device)
            image_features = vision_tower_instance.forward(images)

            predicted_classes = classifier.forward(image_features)

            loss = criterion(predicted_classes, ground_truths)  # Compute loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation Loop
        vision_tower_instance.eval()
        classifier.eval()
        val_loss = 0
        with torch.no_grad():
            for batch, (image_paths, labels_str) in enumerate(val_loader):
                ground_truths = torch.Tensor(convert_label_str(labels_str)).to(device)
                # print("ground_truths:", ground_truths)

                images = vision_tower_instance.preprocess(image_paths).to(device)
                image_features = vision_tower_instance.forward(images)

                predicted_classes = classifier.forward(image_features)

                val_loss += criterion(predicted_classes, ground_truths).item()

        avg_val_loss = val_loss / len(val_loader)
        end_time = time.time()
        epoch_time = end_time - start_time
        print(
            f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f} seconds"
        )

        avg_train_loss_arr.append(avg_train_loss)
        avg_val_loss_arr.append(avg_val_loss)

    # save the models separately

    vision_tower_instance.vision_tower.save_pretrained(save_directory=os.path.join(output_dir,f"vision_tower-epoch-{epochs}-lr-{lr}"))
    vision_tower_instance.image_processor.save_pretrained(save_directory=os.path.join(output_dir, f"image-processor-epoch-{epochs}-lr-{lr}"))

    # torch.save(
    #     vision_tower_instance.state_dict(),
    #     os.path.join(output_dir, f"vision_tower-epoch-{epochs}-lr-{lr}.pth"),
    # )
    torch.save(
        classifier.state_dict(),
        os.path.join(output_dir, f"classifier-epoch-{epochs}-lr-{lr}.pth"),
    )

    return vision_tower_instance, classifier


def test_validation_set(vision_tower_instance, classifier, val_loader):
    '''Generate the y_true and y_pred for validation purposes'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_tower_instance.to(device)
    classifier.to(device)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for _, (image_paths, labels_str) in enumerate(val_loader):
            ground_truths = torch.Tensor(convert_label_str(labels_str)).to(device)
            y_true.append(ground_truths)
            # print("len(ground_truths):", len(ground_truths))
            # print("ground_truths:", ground_truths)

            images = vision_tower_instance.preprocess(image_paths).to(device)
            image_features = vision_tower_instance.forward(images)

            class_probabilty = classifier.forward(image_features)
            y_pred.append(class_probabilty)
            # print("class_probabilty:", class_probabilty)
    
    return y_true, y_pred


def load_classifier(path):
  '''Load existing CLIPDiseaseClassifier'''
  classifier = CLIPDiseaseClassifier()
  state_dict = torch.load(path, weights_only=True)
  classifier.load_state_dict(state_dict)
  print("classifier:", classifier)
  return classifier

def load_existing_vision(path):
  '''Load existing CustomCLIPVisionTower'''
  vision_tower_instance = build_vision_tower(ModelArguments(vision_tower=path))
  vision_tower_instance.load_model()
  print("vision_tower_instance:", vision_tower_instance)
  return vision_tower_instance


if __name__ == "__main__":
  import utils
  classifier = load_classifier("./classifier-epoch-1-lr-0.0001.pth")
  vision_tower_instance = load_existing_vision("./vision_tower-epoch-1-lr-0.0001")

  validation_data = CustomImageDataset(
      annotations_file="/home/r11kaijun/MIMIC-CXR/processed_data/processed_mimic-cxr-2.0.0-chexpert_validate.csv",
      img_dir="/home/r11kaijun/physionet.org/files/mimic-cxr-jpg/2.1.0",
  )
  validation_dataloader = DataLoader(validation_data, batch_size=64, shuffle=True)

  y_true, y_pred = test_validation_set(vision_tower_instance, classifier, validation_dataloader)

  y_pred = utils.get_classification(y_pred).cpu()
  y_true = utils.flatten(y_true).cpu()

  from sklearn import metrics
  accuracy_score = metrics.accuracy_score(y_pred=y_pred, y_true=y_true)
  recall_score = metrics.recall_score(y_pred=y_pred, y_true=y_true)
  precision_score = metrics.precision_score(y_pred=y_pred, y_true=y_true)
  f1_score = metrics.f1_score(y_pred=y_pred, y_true=y_true)

  print("accuracy score:", accuracy_score)
  print("precision score:", precision_score)
  print("recall score:", recall_score)
  print("f1 score:", f1_score)


# # Finetune the clip encoder
# vision_tower = build_vision_tower(ModelArguments())
# classifier = CLIPDiseaseClassifier()
# print("vision_tower:", vision_tower)
# print("classifier:", classifier)

# training_data = CustomImageDataset(
#     annotations_file="/home/r11kaijun/MIMIC-CXR/processed_data/processed_mimic-cxr-2.0.0-chexpert_train.csv",
#     img_dir="/home/r11kaijun/physionet.org/files/mimic-cxr-jpg/2.1.0",
# )
# validation_data = CustomImageDataset(
#     annotations_file="/home/r11kaijun/MIMIC-CXR/processed_data/processed_mimic-cxr-2.0.0-chexpert_validate.csv",
#     img_dir="/home/r11kaijun/physionet.org/files/mimic-cxr-jpg/2.1.0",
# )

# train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
# validation_dataloader = DataLoader(validation_data, batch_size=64, shuffle=True)
# print("validation dataloader:", len(validation_dataloader))

# # finetune the clip encoder on the training dataset 
# train_clip_classifier(
#     vision_tower,
#     classifier,
#     train_loader=train_dataloader,
#     val_loader=validation_dataloader,
#     output_dir=".",
#     epochs=1,
# )