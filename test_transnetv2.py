import os
import json
import torch
import numpy as np
import urllib.request
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any

# Cố gắng import decord, nếu không có thì báo lỗi rõ ràng
try:
    from decord import VideoReader, cpu
except ImportError:
    raise ImportError("Vui lòng cài đặt decord bằng lệnh: pip install decord")

# Hàm TỰ ĐỘNG tải Model và file nguồn (Nếu chưa có)
def ensure_transnet_files():
    py_url = "https://raw.githubusercontent.com/soCzech/TransNetV2/master/inference-pytorch/transnetv2_pytorch.py"
    pth_url = "https://huggingface.co/MiaoshouAI/transnetv2-pytorch-weights/resolve/main/transnetv2-pytorch-weights.pth"
    
    if not os.path.exists("transnetv2_pytorch.py"):
        print("⏳ Đang tự động tải mã nguồn TransNetV2 từ GitHub...")
        urllib.request.urlretrieve(py_url, "transnetv2_pytorch.py")
        print("✅ Đã tải xong transnetv2_pytorch.py")
        
    if not os.path.exists("transnetv2-pytorch-weights.pth"):
        print("⏳ Đang tự động tải tệp Tạ (Weights) của TransNetV2 (khoảng vài chục MB)...")
        urllib.request.urlretrieve(pth_url, "transnetv2-pytorch-weights.pth")
        print("✅ Đã tải xong weights!")

# Gọi hàm tải ngay lập tức
ensure_transnet_files()
from transnetv2_pytorch import TransNetV2


class TransNetDataset(Dataset):
    def __init__(self, video_path: str, window_size: int = 100):
        self.video_path = video_path
        self.window_size = window_size
        
        # Sử dụng tính năng native resize của decord (chạy bằng C++) cực kỳ nhanh!
        # TransNetV2 yêu cầu width=48, height=27
        self.vr = VideoReader(video_path, ctx=cpu(0), width=48, height=27)
        self.total_frames = len(self.vr)
        self.fps = self.vr.get_avg_fps()
        
        self.num_windows = int(np.ceil(self.total_frames / self.window_size))
        
    def __len__(self):
        return self.num_windows
        
    def __getitem__(self, idx):
        start_frame = idx * self.window_size
        end_frame = min(start_frame + self.window_size, self.total_frames)
        
        frame_indices = list(range(start_frame, end_frame))
        frames = self.vr.get_batch(frame_indices).asnumpy()
        
        actual_len = frames.shape[0]
        pad_len = self.window_size - actual_len
        
        if pad_len > 0:
            last_frame = frames[-1:]
            pad_frames = np.repeat(last_frame, pad_len, axis=0)
            frames = np.concatenate([frames, pad_frames], axis=0)
            
        # decord đã resize sẵn, ta chỉ việc chuyển sang tensor
        # TransNetV2 pytorch expects [B, T, 27, 48, 3] of type torch.uint8!
        tensor_frames = torch.from_numpy(frames)
        
        return tensor_frames, actual_len

def process_predictions(predictions: np.ndarray, hard_threshold: float = 0.5) -> List[Dict[str, Any]]:
    predictions = (predictions > hard_threshold).astype(np.uint8)
    scenes = []
    t_prev, start = 0, 0
    
    for i, pred in enumerate(predictions):
        if pred == 1 and t_prev == 0:
            scenes.append({
                "start_frame": start,
                "end_frame": i,
                "type": "hard" if i - start > 1 else "soft" 
            })
            start = i
        t_prev = pred
        
    if start < len(predictions):
        scenes.append({
            "start_frame": start,
            "end_frame": len(predictions) - 1,
            "type": "hard"
        })
        
    return scenes

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test TransNetV2 Shot Boundary Detection")
    parser.add_argument("--video", type=str, default="sample.mp4", help="Đường dẫn tới video cần test")
    parser.add_argument("--output", type=str, default="transnet_shots.jsonl", help="File xuất kết quả")
    args = parser.parse_args()
    
    # TẮT num_workers (=0) để decord không bị kẹt luồng (deadlock) khi đọc file
    num_workers = 0 
    video_path = args.video
    output_jsonl = args.output
    
    if not os.path.exists(video_path):
        print(f"❌ Không tìm thấy video {video_path}")
        return
        
    # Tăng batch_size lên 16 để tận dụng sức mạnh tính toán song song của GPU T4
    batch_size = 16
    print(f"Khởi tạo DataLoader với batch_size={batch_size}...")
    dataset = TransNetDataset(video_path, window_size=100)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang chạy Inference trên thiết bị: {device}")
    
    model = TransNetV2()
    # Tự động nạp bộ weights xịn vừa tải về (Hỗ trợ nạp trên máy tính không có GPU)
    model.load_state_dict(torch.load("transnetv2-pytorch-weights.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    all_preds = []
    
    with torch.no_grad():
        for batch_idx, (frames, actual_lens) in enumerate(dataloader):
            frames = frames.to(device)
            
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits, _ = model(frames)
                
            probs = torch.sigmoid(logits).cpu().numpy()
            
            # Xử lý theo từng batch
            for b in range(probs.shape[0]):
                actual_len = actual_lens[b].item()
                valid_probs = probs[b, :actual_len, 0] 
                all_preds.extend(valid_probs.tolist())
            
            if (batch_idx + 1) % max(1, (10 // batch_size)) == 0:
                print(f"Đã xử lý {min((batch_idx + 1) * batch_size, len(dataset))}/{len(dataset)} windows...")
                
    print("Tiến hành phân tích kết quả dự đoán (Post-processing)...")
    all_preds_np = np.array(all_preds)
    fps = dataset.fps
    
    shots = process_predictions(all_preds_np, hard_threshold=0.5)
    
    print(f"Đang ghi kết quả ra {output_jsonl}...")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for idx, shot in enumerate(shots):
            shot_id = idx
            start_f = int(shot["start_frame"])
            end_f = int(shot["end_frame"])
            
            record = {
                "shot_id": shot_id,
                "start_frame": start_f,
                "end_frame": end_f,
                "start_timestamp": round(start_f / fps, 3),
                "end_timestamp": round(end_f / fps, 3),
                "type": shot["type"]
            }
            f.write(json.dumps(record) + "\n")
            
    print(f"✅ Hoàn thành! Tìm thấy {len(shots)} shots. Dữ liệu đã lưu tại {output_jsonl}")

if __name__ == "__main__":
    main()
