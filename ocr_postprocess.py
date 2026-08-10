"""
OCR Post-Processor – Sửa lỗi chính tả tiếng Việt cho Global OCR
================================================================
Sử dụng Qwen2.5-1.5B-Instruct (local, ~3GB VRAM) để sửa lỗi ngữ pháp
và dấu tiếng Việt trong text OCR trích xuất từ keyframe.

CHỈ dùng cho Global OCR (shot-level), KHÔNG dùng cho Object OCR.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "Bạn là trợ lý sửa lỗi chính tả tiếng Việt cho kết quả OCR. "
    "Nhiệm vụ: sửa lỗi chính tả, dấu thanh, và lỗi nhận dạng ký tự sai. "
    "Quy tắc bắt buộc:\n"
    "- CHỈ sửa lỗi, KHÔNG thêm bớt nội dung.\n"
    "- Giữ nguyên tên riêng, số liệu, ký hiệu.\n"
    "- Nếu text đầu vào đã đúng, trả về nguyên văn.\n"
    "- Nếu text rỗng hoặc vô nghĩa hoàn toàn, trả về chuỗi rỗng.\n"
    "- Chỉ trả về đoạn text đã sửa, không giải thích gì thêm."
)

USER_TEMPLATE = "Sửa lỗi OCR tiếng Việt:\n{text}"


class OCRPostProcessor:
    """Local LLM-based OCR error corrector using Qwen2.5-1.5B-Instruct."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 device: str = "cuda", fp16: bool = True,
                 max_new_tokens: int = 256):
        self.device = device
        self.max_new_tokens = max_new_tokens

        print(f"[OCR PostProcessor] Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if fp16 and device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(device)
        self.model.eval()
        print(f"[OCR PostProcessor] Ready on {device}.")

    def correct(self, ocr_text: str) -> str:
        """
        Sửa lỗi chính tả cho một đoạn OCR text.
        Trả về text đã sửa hoặc text gốc nếu không cần sửa.
        """
        if not ocr_text or len(ocr_text.strip()) < 3:
            return ocr_text

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(text=ocr_text)}
        ]

        try:
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                input_text, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,       # Greedy — deterministic
                    temperature=1.0,
                    repetition_penalty=1.1
                )

            # Decode only the newly generated tokens
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            result = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Sanity check: if LLM output is empty or way longer than input, keep original
            if not result or len(result) > len(ocr_text) * 3:
                return ocr_text

            return result

        except Exception as e:
            print(f"[OCR PostProcessor] Error: {e}. Returning original text.")
            return ocr_text

    def correct_batch(self, texts: list) -> list:
        """Sửa lỗi cho danh sách text. Dùng cho batch processing shots."""
        return [self.correct(t) for t in texts]
