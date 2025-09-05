# # import torch
# # import torch.nn.functional as F
# # from PIL import Image
# # from open_clip import create_model_from_pretrained, get_tokenizer
# # from torchvision import transforms
# # import matplotlib.pyplot as plt
# # import numpy as np
# # import warnings
# # import logging
# # import os
# # from pathlib import Path
# # from crewai import Agent
# # from pydantic import PrivateAttr



# # class TumourPredictor:
# #     def __init__(self, checkpoint_path, device=None):
# #         """
# #         Initializes BioMedCLIP with your fine-tuned weights for binary tumour classification.
# #         """
# #         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
# #         # Load BioMedCLIP
# #         self.model, self.preprocess = create_model_from_pretrained(
# #             'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
# #         )
# #         self.model.to(self.device)
# #         self.model.eval()

# #         # Load tokenizer
# #         self.tokenizer = get_tokenizer(
# #             'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
# #         )

# #         # Load fine-tuned weights
# #         state_dict = torch.load(checkpoint_path, map_location=self.device)
# #         self.model.load_state_dict(state_dict, strict=False)

# #         # Initialize text prompts (same as used in training)
# #         self.class_names = ['no tumour', 'tumour']
# #         text_prompts = [f"This image of breast tissue contains {label}" for label in self.class_names]

# #         with torch.no_grad():
# #             tokenized = self.tokenizer(text_prompts).to(self.device)
# #             self.text_features = self.model.encode_text(tokenized)
# #             self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

# #         # Temperature (you trained it as a parameter)
# #         self.temperature = torch.tensor(1.0).to(self.device)  # If you saved temperature, load it instead.

# #     def predict(self, image_path):
# #         """
# #         Predict tumour or no tumour for a single image.
# #         """
# #         image = Image.open(image_path).convert("RGB")
# #         image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

# #         with torch.no_grad():
# #             image_features = self.model.encode_image(image_tensor)
# #             image_features = image_features / image_features.norm(dim=-1, keepdim=True)

# #             logits = (image_features @ self.text_features.T) * self.temperature
# #             probs = F.softmax(logits, dim=-1).cpu().numpy().flatten()

# #         pred_idx = probs.argmax()
# #         return {"label": self.class_names[pred_idx], "confidence": float(probs[pred_idx])}


# # # from tumour_predictor import TumourPredictor

# # class TumourAgent(Agent):
# #     _predictor: TumourPredictor = PrivateAttr()

# #     def __init__(self, checkpoint_path="biomedclip_finetuned.pth"):
# #         super().__init__(
# #             role="Medical Image Classifier",
# #             goal="Classify breast tissue images as tumour or no tumour.",
# #             backstory=(
# #                 "An AI model fine-tuned on BioMedCLIP to detect breast cancer "
# #                 "tumour presence with high accuracy."
# #             )
# #         )
# #         self._predictor = TumourPredictor(checkpoint_path)

# #     def run(self, image_path):
# #         """Analyze the image and return tumour prediction."""
# #         result = self._predictor.predict(image_path)
# #         label = result["label"]
# #         confidence = result["confidence"]
# #         return f"The image is classified as '{label}' with {confidence:.2f} confidence."


# # def run_tumour_agent(image_path, checkpoint="C:/Users/arish/Downloads/finetuned_biomedclip.pth"):
# #     agent = TumourAgent(checkpoint)
# #     result = agent._predictor.predict(image_path)  # dict {label, confidence}
# #     return result

# import torch
# import torch.nn.functional as F
# from PIL import Image
# from open_clip import create_model_from_pretrained, get_tokenizer
# from torchvision import transforms
# import matplotlib.pyplot as plt
# import numpy as np
# import warnings
# import logging
# import os
# from pathlib import Path
# from crewai import Agent
# from pydantic import PrivateAttr



# class TumourPredictor:
#     def __init__(self, checkpoint_path, device=None):
#         """
#         Initializes BioMedCLIP with your fine-tuned weights for binary tumour classification.
#         """
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
#         # Load BioMedCLIP
#         self.model, self.preprocess = create_model_from_pretrained(
#             'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
#         )
#         self.model.to(self.device)
#         self.model.eval()

#         # Load tokenizer
#         self.tokenizer = get_tokenizer(
#             'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
#         )

#         if checkpoint_path and os.path.exists(checkpoint_path):
#             # Load fine-tuned weights
#             state_dict = torch.load(checkpoint_path, map_location=self.device)
#             self.model.load_state_dict(state_dict, strict=False)
#             print(f"✅ Loaded fine-tuned weights from {checkpoint_path}")
#         else:
#             print(f"⚠️ Warning: Checkpoint not found at {checkpoint_path}, using base model")

#         # Initialize text prompts (same as used in training)
#         self.class_names = ['no tumour', 'tumour']
#         text_prompts = [f"This image of breast tissue contains {label}" for label in self.class_names]

#         with torch.no_grad():
#             tokenized = self.tokenizer(text_prompts).to(self.device)
#             self.text_features = self.model.encode_text(tokenized)
#             self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

#         # Temperature (you trained it as a parameter)
#         self.temperature = torch.tensor(1.0).to(self.device)  # If you saved temperature, load it instead.

#     def predict_from_file(self, image_path):
#         """Predict from file path."""
#         image = Image.open(image_path).convert("RGB")
#         return self._predict_image(image)
    
#     def predict_from_image(self, pil_image):
#         """Predict from PIL Image object."""
#         return self._predict_image(pil_image)
    
#     def _predict_image(self, image):
#         """Internal method to predict from PIL Image."""
#         image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

#         with torch.no_grad():
#             image_features = self.model.encode_image(image_tensor)
#             image_features = image_features / image_features.norm(dim=-1, keepdim=True)

#             logits = (image_features @ self.text_features.T) * self.temperature
#             probs = F.softmax(logits, dim=-1).cpu().numpy().flatten()

#         pred_idx = probs.argmax()
#         return {"label": self.class_names[pred_idx], "confidence": float(probs[pred_idx])}

#     def predict(self, image_path):
#         """Predict tumour or no tumour for a single image."""
#         return self.predict_from_file(image_path)


# # from tumour_predictor import TumourPredictor

# class TumourAgent(Agent):
#     _predictor: TumourPredictor = PrivateAttr()

#     def __init__(self, checkpoint_path="C:/Users/arish/Downloads/finetuned_biomedclip.pth"):
#         super().__init__(
#             role="Medical Image Classifier",
#             goal="Classify breast tissue images as tumour or no tumour.",
#             backstory=(
#                 "An AI model fine-tuned on BioMedCLIP to detect breast cancer "
#                 "tumour presence with high accuracy."
#             )
#         )
#         self._predictor = TumourPredictor(checkpoint_path)

#     def run(self, image_path):
#         """Analyze the image and return tumour prediction."""
#         result = self._predictor.predict(image_path)
#         return result
#         # label = result["label"]
#         # confidence = result["confidence"]
#         # return f"The image is classified as '{label}' with {confidence:.2f} confidence."


# def run_tumour_agent(image_path, checkpoint="C:/Users/arish/Downloads/finetuned_biomedclip.pth"):
#     try:
#         agent = TumourAgent(checkpoint)
#         result = agent._predictor.predict(image_path)  # dict {label, confidence}
#         return result
#     except Exception as e:
#         print(f"⚠️ Error in tumour agent: {e}")
#         # Return default prediction if model fails
#         return {"label": "tumour", "confidence": 0.5}

import torch
import torch.nn.functional as F
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer
import os
from crewai.tools import tool   # <-- use CrewAI’s tool decorator


class TumourPredictor:
    def __init__(self, checkpoint_path, device=None):
        """
        Initializes BioMedCLIP with your fine-tuned weights for binary tumour classification.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load BioMedCLIP
        self.model, self.preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.model.to(self.device)
        self.model.eval()

        # Load tokenizer
        self.tokenizer = get_tokenizer(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )

        if checkpoint_path and os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded fine-tuned weights from {checkpoint_path}")
        else:
            print(f"⚠️ Warning: Checkpoint not found at {checkpoint_path}, using base model")

        # Initialize text prompts
        self.class_names = ['no tumour', 'tumour']
        text_prompts = [f"This image of breast tissue contains {label}" for label in self.class_names]

        with torch.no_grad():
            tokenized = self.tokenizer(text_prompts).to(self.device)
            self.text_features = self.model.encode_text(tokenized)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

        self.temperature = torch.tensor(1.0).to(self.device)

    def _predict_image(self, image):
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = (image_features @ self.text_features.T) * self.temperature
            probs = F.softmax(logits, dim=-1).cpu().numpy().flatten()

        pred_idx = probs.argmax()
        return {"label": self.class_names[pred_idx], "confidence": float(probs[pred_idx])}

    def predict(self, image_path):
        """Predict tumour or no tumour for a single image."""
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        return self._predict_image(image)


# ✅ Tool wrapper around the predictor
@tool("tumour_classifier_tool")
def tumour_classifier_tool(image_path: str) -> dict:
    """
    Classify a histopathology breast image as tumour or no tumour.
    Args:
        image_path: Path to the image file
    Returns:
        dict: {"label": "tumour"/"no tumour", "confidence": float}
    """
    try:
        checkpoint = "C:/Users/arish/Downloads/finetuned_biomedclip.pth"
        predictor = TumourPredictor(checkpoint)
        return predictor.predict(image_path)
    except Exception as e:
        print(f"⚠️ Error in tumour classifier tool: {e}")
        return {"label": "tumour", "confidence": 0.5}
