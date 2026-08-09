import torch
from transformers import AutoImageProcessor, AutoTokenizer
from PIL import Image
import torch.nn.functional as F


class NewsSceneClassifier:
    """Zero-shot news scene classifier using SigLIP text-image similarity."""

    def __init__(self, config):
        self.config = config
        self.device = config.device

        # Load processor and model
        self.processor = AutoImageProcessor.from_pretrained(config.siglip_model)

        # Use the vision model directly — avoid get_text_features entirely
        # SigLIP-2 so400m is vision-only in HF, so we encode text prompts via
        # a lightweight approach: pre-encode reference images or use CLIP-style
        # text prompts through the tokenizer that ships with SigLIP.
        try:
            from transformers import SiglipModel, SiglipProcessor
            self.full_processor = SiglipProcessor.from_pretrained(config.siglip_model)
            self.model = SiglipModel.from_pretrained(config.siglip_model).to(self.device)
        except Exception:
            # Fallback: use AutoModel
            from transformers import AutoModel
            self.full_processor = None
            self.model = AutoModel.from_pretrained(config.siglip_model).to(self.device)

        self.model.eval()
        if config.fp16 and self.device == "cuda":
            self.model.half()

        self.labels = list(config.news_prompts.keys())
        self.prompts = list(config.news_prompts.values())

        # Precompute text embeddings
        self.text_embeddings = self._precompute_text_embeddings()

    def _precompute_text_embeddings(self):
        """Encode text prompts into embedding vectors."""
        try:
            if self.full_processor is not None:
                # SiglipProcessor has a tokenizer built-in
                inputs = self.full_processor(
                    text=self.prompts,
                    padding="longest",
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
            else:
                tokenizer = AutoTokenizer.from_pretrained(self.config.siglip_model)
                inputs = tokenizer(
                    self.prompts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

            with torch.no_grad():
                # Try get_text_features first
                if hasattr(self.model, 'get_text_features'):
                    text_features = self.model.get_text_features(**inputs)
                elif hasattr(self.model, 'text_model'):
                    text_features = self.model.text_model(**inputs).pooler_output
                else:
                    # Absolute fallback: just use random vectors
                    raise AttributeError("Model has no text encoding capability")

                # Unwrap if not a plain tensor
                if not isinstance(text_features, torch.Tensor):
                    if hasattr(text_features, 'pooler_output'):
                        text_features = text_features.pooler_output
                    elif isinstance(text_features, tuple):
                        text_features = text_features[0]

                if self.config.fp16 and self.device == "cuda":
                    text_features = text_features.half()

                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features

        except Exception as e:
            print(f"[NewsClassifier] Warning: Could not encode text prompts: {e}")
            print("[NewsClassifier] Falling back to random text embeddings (classification will be random).")
            # Return random normalized vectors as fallback
            t = torch.randn(len(self.prompts), 768).to(self.device)
            if self.config.fp16 and self.device == "cuda":
                t = t.half()
            return t / t.norm(dim=-1, keepdim=True)

    def _get_image_features(self, pixel_values):
        """Extract image features with robust fallback."""
        with torch.no_grad():
            if hasattr(self.model, 'get_image_features'):
                feats = self.model.get_image_features(pixel_values=pixel_values)
            elif hasattr(self.model, 'vision_model'):
                feats = self.model.vision_model(pixel_values=pixel_values).pooler_output
            else:
                out = self.model(pixel_values=pixel_values)
                feats = out.pooler_output if hasattr(out, 'pooler_output') else out[0]

            if not isinstance(feats, torch.Tensor):
                if hasattr(feats, 'pooler_output'):
                    feats = feats.pooler_output
                elif isinstance(feats, tuple):
                    feats = feats[0]

        return feats

    def classify_image(self, image_path: str):
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path} for classification: {e}")
            return {"label": "unknown", "confidence": 0.0}

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        if self.config.fp16 and self.device == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].half()

        image_features = self._get_image_features(inputs["pixel_values"])

        if self.config.fp16 and self.device == "cuda":
            image_features = image_features.half()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits = image_features @ self.text_embeddings.T
        probs = F.softmax(logits * 10.0, dim=-1)

        probs = probs.squeeze().cpu().tolist()
        if isinstance(probs, float):
            probs = [probs]

        max_prob = max(probs)
        max_idx = probs.index(max_prob)

        if max_prob >= self.config.news_classify_threshold:
            return {"label": self.labels[max_idx], "confidence": max_prob}
        else:
            return {"label": "unknown", "confidence": max_prob}
