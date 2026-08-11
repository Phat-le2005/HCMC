"""
OCR Post-Processor – Sửa lỗi chính tả tiếng Việt cho Global OCR
================================================================
Sử dụng Qwen2.5-1.5B-Instruct (local, ~3GB VRAM) để sửa lỗi ngữ pháp
và dấu tiếng Việt trong text OCR trích xuất từ keyframe.

CHỈ dùng cho Global OCR (shot-level), KHÔNG dùng cho Object OCR.
"""

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ═══════════════════════════════════════════════════════════════════
# Pre-clean: Regex filters for broadcast overlay noise
# ═══════════════════════════════════════════════════════════════════

# Broadcast timestamp patterns: "06:30:14", "06:31:04", "6:1", etc.
# Also catches malformed timestamps like "06:32:621", "06:32.21"
RE_TIMESTAMP = re.compile(r'\b\d{1,2}[:.]\d{2}(?:[:.]\d{1,3})?\b')

# Channel logo / watermark fragments commonly misread by OCR
# Matches: HD, HTV, HTVS, HTVQ, HTVX, HTM, HTMS, HTZ, HTL, HTA, HTD,
#          HIV, HIVS, HIVQ, HIVE, HIVER, HIL, HMQ, HT (standalone),
#          GID, ND, MD, AD, ZHD, Zhd, Zad, Zap, Khp, Phd, Vhd, Chd, etc.
RE_CHANNEL_LOGO = re.compile(
    r'\b(?:'
    r'H?TV[SQX]?'           # HTV, HTVS, HTVQ, HTVX, TV
    r'|H?IV[ESQ]?R?'        # HIV, HIVS, HIVE, HIVER, HIVQ
    r'|HTM[S]?'             # HTM, HTMS
    r'|HT[ZDLA]?'           # HT, HTZ, HTD, HTL, HTA
    r'|HIL|HMQ|HOV|HP'      # HIL, HMQ, HOV, HP
    r'|HD'                  # HD
    r'|[ZKVCP]?[Hh]d'       # ZHD, Zhd, Vhd, Chd, Khd, Phd
    r'|Khp|Phd|Vhd|Chd'     # Misread prefix+logo combos
    r'|Zap|Zad'             # Misread logo prefix combos
    r'|GID|ND|MD|AD'        # Other junk abbreviations
    r'|GIVE|Inse|GMT'       # English junk from misread overlays
    r')\b',
    re.IGNORECASE
)

# Broadcast label text: "TINCHÍNH", "Tin chinh", "Tin chính"
RE_BROADCAST_LABEL = re.compile(r'\bTIN\s*CH[IÍ]NH\b', re.IGNORECASE)

# Standalone short junk tokens from broadcast overlays
RE_JUNK_NUMBERS = re.compile(r'\b(?:19|00|60|69|66|160|10)\b')

# The word "giây" (second) from TV clock overlays
RE_CLOCK_WORD = re.compile(r'\bgiây\b', re.IGNORECASE)

# Repeated whitespace
RE_MULTI_SPACE = re.compile(r'\s{2,}')


def pre_clean_ocr(text: str) -> str:
    """
    Remove broadcast overlay noise from raw OCR text BEFORE LLM correction.
    This handles watermarks (HTV, HD, HIVS...) and on-screen timestamps.
    """
    if not text:
        return text
    
    # Step 1: Remove timestamps (e.g., "06:30:14")
    text = RE_TIMESTAMP.sub('', text)
    
    # Step 2: Remove channel logo fragments
    text = RE_CHANNEL_LOGO.sub('', text)
    
    # Step 3: Remove broadcast labels ("TINCHÍNH", "Tin chính")
    text = RE_BROADCAST_LABEL.sub('', text)
    
    # Step 4: Remove common standalone junk numbers from overlays
    text = RE_JUNK_NUMBERS.sub('', text)
    
    # Step 4: Remove stray "giây" from clock overlay
    text = RE_CLOCK_WORD.sub('', text)
    
    # Step 5: Collapse multiple spaces and strip
    text = RE_MULTI_SPACE.sub(' ', text).strip()
    
    # Step 6: Remove dangling punctuation/symbols left behind
    text = re.sub(r'(?:^[\s,.\-:;/()]+|[\s,.\-:;/()]+$)', '', text)
    
    return text


SYSTEM_PROMPT = (
    "Bạn là trợ lý sửa lỗi chính tả tiếng Việt cho kết quả OCR từ video thời sự. "
    "Nhiệm vụ: sửa lỗi chính tả, dấu thanh, và lỗi nhận dạng ký tự sai. "
    "Quy tắc bắt buộc:\n"
    "- CHỈ sửa lỗi, KHÔNG thêm bớt nội dung.\n"
    "- Giữ nguyên tên riêng, địa danh, số liệu, ký hiệu.\n"
    "- Nếu gặp từ viết tắt vô nghĩa hoặc nhiễu logo đài (ví dụ: HTV, HIVS, HD), hãy XÓA chúng.\n"
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
        Pipeline: pre_clean (regex) → Qwen LLM (grammar fix).
        """
        if not ocr_text or len(ocr_text.strip()) < 3:
            return ocr_text

        # Step 1: Regex pre-clean to remove broadcast noise
        cleaned = pre_clean_ocr(ocr_text)
        
        if not cleaned or len(cleaned.strip()) < 3:
            return ""

        # Step 2: LLM correction
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(text=cleaned)}
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

            # Sanity check: if LLM output is empty or way longer than input, keep cleaned
            if not result or len(result) > len(cleaned) * 3:
                return cleaned

            return result

        except Exception as e:
            print(f"[OCR PostProcessor] Error: {e}. Returning pre-cleaned text.")
            return cleaned

    def correct_batch(self, texts: list) -> list:
        """Sửa lỗi cho danh sách text. Dùng cho batch processing shots."""
        return [self.correct(t) for t in texts]

