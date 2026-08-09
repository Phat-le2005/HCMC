import torch
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import torch.nn.functional as F

class NewsSceneClassifier:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.processor = AutoImageProcessor.from_pretrained(config.siglip_model)
        self.model = AutoModel.from_pretrained(config.siglip_model).to(self.device)
        self.model.eval()
        if config.fp16 and self.device == "cuda":
            self.model.half()
            
        self.labels = list(config.news_prompts.keys())
        self.prompts = list(config.news_prompts.values())
        
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.siglip_model)
        
        # Precompute text embeddings for zero-shot
        self.text_embeddings = self._precompute_text_embeddings()

    def _precompute_text_embeddings(self):
        inputs = self.tokenizer(self.prompts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            if not isinstance(text_features, torch.Tensor):
                text_features = text_features[0] if isinstance(text_features, tuple) else text_features.pooler_output
                
            if self.config.fp16 and self.device == "cuda":
                text_features = text_features.half()
            # Normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def classify_image(self, image_path: str):
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path} for classification: {e}")
            return {"label": "unknown", "confidence": 0.0}
            
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        if self.config.fp16 and self.device == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].half()
            
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            if not isinstance(image_features, torch.Tensor):
                image_features = image_features[0] if isinstance(image_features, tuple) else image_features.pooler_output
                
            if self.config.fp16 and self.device == "cuda":
                image_features = image_features.half()
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # cosine similarity as logits
            logits = image_features @ self.text_embeddings.T
            probs = F.softmax(logits * 10.0, dim=-1) # scaling factor
            
            probs = probs.squeeze().cpu().tolist()
            
        max_prob = max(probs)
        max_idx = probs.index(max_prob)
        
        if max_prob >= self.config.news_classify_threshold:
            return {"label": self.labels[max_idx], "confidence": max_prob}
        else:
            return {"label": "unknown", "confidence": max_prob}
