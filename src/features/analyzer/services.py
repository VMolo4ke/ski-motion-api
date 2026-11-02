import os
import cv2
import numpy as np
import modal
from features.analyzer.schemas import AnalysisResponse, FrameAnalysis, Keypoint

# Собираем контейнер (без CUDA, так как работаем на CPU)
image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("opencv-python-headless", "numpy", "ultralytics", "pydantic")
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App("video-inference-serverless")

@app.function(
    image=image, 
    timeout=300,
    env={"PYTHONPATH": "/root/src:/root"} # Чтобы Modal внутри облака видел ваши schemas.py
)
def predict_pose_on_cpu(video_bytes: bytes) -> tuple[list, list]:
    """Выполняется в облаке Modal на CPU"""
    import cv2
    from ultralytics import YOLO
    
    model = YOLO("yolov8m-pose.pt") 
    temp_video_path = "/tmp/remote_input_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(video_bytes)
        
    # Считаем таймстампы прямо в облаке для 100% синхронизации с кадрами YOLO
    cap = cv2.VideoCapture(temp_video_path)
    timestamps_ms = []
    while cap.isOpened():
        ret, _ = cap.read()
        if not ret:
            break
        timestamps_ms.append(int(cap.get(cv2.CAP_PROP_POS_MSEC)))
    cap.release()
    
    results = model(temp_video_path, device="cpu", verbose=False)
    
    raw_timeline = []
    for res in results:
        if res.keypoints is not None and len(res.keypoints.data) > 0:
            kpts = res.keypoints.data.cpu().numpy().tolist()
            raw_timeline.append(kpts) # Передаем список людей с их точками
        else:
            raw_timeline.append([])
            
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
        
    return raw_timeline, timestamps_ms


def run_large_analysis(video_path: str) -> AnalysisResponse:
    """Оркестрирует инференс и делает EMA-сглаживание"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео по пути: {video_path}")
        
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    with open(video_path, "rb") as f:
        video_bytes = f.read()

    try:
        # ИСПРАВЛЕНО: Обернули вызов в контекстный менеджер app.run() для гидратации метаданных в Modal
        with app.run():
            raw_timeline, timestamps_ms = predict_pose_on_cpu.remote(video_bytes)
    except Exception as e:
        raise RuntimeError(f"Ошибка инференса в Modal: {e}")

    POINTS = {
        "L_shoulder": 5, "R_shoulder": 6, "L_elbow": 7, "R_elbow": 8,
        "L_wrist": 9, "R_wrist": 10, "L_hip": 11, "R_hip": 12,
        "L_knee": 13, "R_knee": 14, "L_ankle": 15, "R_ankle": 16
    }
    
    timeline_frames = []
    prev_kpts = {name: None for name in POINTS}
    alpha = 0.4

    for frame_idx, raw_kpts in enumerate(raw_timeline):
        timestamp_ms = timestamps_ms[frame_idx] if frame_idx < len(timestamps_ms) else frame_idx * 33
        smoothed_points = {}
        
        # Проверяем, что в кадре найден хотя бы один человек
        if len(raw_kpts) > 0:
            person_kpts = raw_kpts[0] # Берем координаты первого найденного скелета
            
            for name, idx in POINTS.items():
                if idx < len(person_kpts):
                    pt_data = person_kpts[idx] # Список из 3 элементов: [x, y, confidence]
                    
                    # ИСПРАВЛЕНО: Корректно извлекаем координаты x, y и уверенность по индексам
                    pt = np.array([pt_data[0], pt_data[1]])
                    confidence = float(pt_data[2])
                    
                    if pt.any() and confidence > 0.2:
                        if prev_kpts[name] is not None:
                            prev_kpts[name] = prev_kpts[name] * (1 - alpha) + pt * alpha
                        else:
                            prev_kpts[name] = pt
                        
                        smoothed_points[name] = Keypoint(
                            x=float(prev_kpts[name][0] / w),
                            y=float(prev_kpts[name][1] / h),
                            confidence=confidence
                        )
                    elif prev_kpts[name] is not None:
                        # Если трекинг потерян, отдаем прошлую позицию с нулевой уверенностью
                        smoothed_points[name] = Keypoint(
                            x=float(prev_kpts[name][0] / w),
                            y=float(prev_kpts[name][1] / h),
                            confidence=0.0
                        )
        
        timeline_frames.append(FrameAnalysis(time=timestamp_ms, points=smoothed_points))

    return AnalysisResponse(frames=timeline_frames)
