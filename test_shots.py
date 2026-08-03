import cv2
import os
from pathlib import Path
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector, ThresholdDetector

def test_shot_detection(video_path: str, method: str = "content", threshold: float = 27.0, min_scene_len: int = 15):
    """
    Test tách shot trên máy local với các phương pháp khác nhau.
    Các method hỗ trợ: 
    - "content": Dùng ContentDetector (Mặc định, tốt nhất cho video thông thường)
    - "adaptive": Dùng AdaptiveDetector (Tốt cho video có nhiều chuyển động nhanh/sáng mờ)
    - "threshold": Dùng ThresholdDetector (Dành cho các shot chuyển cảnh qua màn hình đen/trắng)
    """
    print(f"\n[{method.upper()}] Đang test video: {video_path} với threshold = {threshold}")
    
    video = open_video(video_path)
    scene_manager = SceneManager()
    
    # Chọn phương pháp tách shot
    if method == "content":
        detector = ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
    elif method == "adaptive":
        detector = AdaptiveDetector(adaptive_threshold=threshold, min_scene_len=min_scene_len)
    elif method == "threshold":
        detector = ThresholdDetector(threshold=threshold, min_scene_len=min_scene_len)
    else:
        raise ValueError(f"Method '{method}' không được hỗ trợ!")

    scene_manager.add_detector(detector)
    scene_manager.detect_scenes(video, show_progress=True)
    
    scene_list = scene_manager.get_scene_list()
    
    print(f"✅ Đã tìm thấy {len(scene_list)} shots!")
    
    # In ra 5 shot đầu tiên để kiểm tra thử
    for i, scene in enumerate(scene_list[:5]):
        print(f"  Shot {i+1}: Frame {scene[0].get_frames()} -> {scene[1].get_frames()} "
              f"(Thời gian: {scene[0].get_timecode()} -> {scene[1].get_timecode()})")
        
    if len(scene_list) > 5:
        print(f"  ... (còn {len(scene_list) - 5} shots nữa)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Baseline Shot Detection (PySceneDetect)")
    parser.add_argument("--video", type=str, required=True, help="Đường dẫn tới video")
    parser.add_argument("--method", type=str, default="content", choices=["content", "adaptive", "threshold"], help="Phương pháp tách shot")
    parser.add_argument("--threshold", type=float, default=27.0, help="Ngưỡng nhạy của thuật toán")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ Không tìm thấy file {args.video}.")
    else:
        test_shot_detection(args.video, method=args.method, threshold=args.threshold)
