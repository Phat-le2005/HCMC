import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image


class NewsSceneClassifier:
    """Zero-shot news scene classifier using SigLIP-2 text-image similarity.
    
    Uses AutoProcessor to jointly encode text prompts + images,
    then computes sigmoid similarity (NOT softmax) as SigLIP uses sigmoid loss.
    """

    def __init__(self, config):
        self.config = config
        self.device = config.device

        print(f"[NewsClassifier] Loading {config.siglip_model}...")
        
        # AutoProcessor handles BOTH text tokenization AND image preprocessing
        self.processor = AutoProcessor.from_pretrained(config.siglip_model)
        self.model = AutoModel.from_pretrained(config.siglip_model).to(self.device)
        self.model.eval()
        
        if config.fp16 and self.device == "cuda":
            self.model.half()

        self.labels = list(config.news_prompts.keys())
        self.prompts = list(config.news_prompts.values())
        
        print(f"[NewsClassifier] Ready. {len(self.labels)} categories: {self.labels}")

    def classify_image(self, image_path: str):
        """Classify a single keyframe image into a news scene type.
        
        Returns: {"label": str, "confidence": float}
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[NewsClassifier] Error loading image {image_path}: {e}")
            return {"label": "unknown", "confidence": 0.0}

        try:
            # Process text + image together through AutoProcessor
            inputs = self.processor(
                text=self.prompts,
                images=image,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)
            
            if self.config.fp16 and self.device == "cuda":
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].half()

            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # SigLIP uses sigmoid (independent probabilities per class)
                logits_per_image = outputs.logits_per_image  # shape: (1, num_labels)
                probs = torch.sigmoid(logits_per_image).squeeze().cpu().tolist()

            if isinstance(probs, float):
                probs = [probs]

            max_prob = max(probs)
            max_idx = probs.index(max_prob)

            if max_prob >= self.config.news_classify_threshold:
                return {"label": self.labels[max_idx], "confidence": max_prob}
            else:
                return {"label": "unknown", "confidence": max_prob}
                
        except Exception as e:
            print(f"[NewsClassifier] Classification error: {e}")
            return {"label": "unknown", "confidence": 0.0}

