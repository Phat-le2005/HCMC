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
    # Thay đường dẫn tới video local của bạn ở đây
    LOCAL_VIDEO_PATH = "sample.mp4" 
    
    if not os.path.exists(LOCAL_VIDEO_PATH):
        print(f"❌ Không tìm thấy file {LOCAL_VIDEO_PATH}. Bạn nhớ đổi tên file video nhé!")
    else:
        # Bạn có thể thử nghiệm nhiều thông số khác nhau cùng 1 lúc:
        
        # 1. Phương pháp mặc định (ContentDetector) - Threshold thấp = cắt nhiều, cao = cắt ít
        test_shot_detection(LOCAL_VIDEO_PATH, method="content", threshold=27.0)
        
        # 2. Phương pháp ContentDetector với ngưỡng nhạy hơn
        # test_shot_detection(LOCAL_VIDEO_PATH, method="content", threshold=20.0)
        
        # 3. Phương pháp Adaptive (thích ứng theo khung hình) - Tốt cho camera rung lắc
        # test_shot_detection(LOCAL_VIDEO_PATH, method="adaptive", threshold=3.0)
