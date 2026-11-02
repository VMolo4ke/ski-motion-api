import os
import shutil
import uuid
import json
import time
import traceback
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, status
from redis import Redis

from features.analyzer.services import run_large_analysis
from features.analyzer.schemas import AnalysisResponse

router = APIRouter(prefix="/analyzer", tags=["YOLO Pose Analyzer"])

REDIS_URL = os.getenv("REDIS_URL")
redis_client = Redis.from_url(REDIS_URL)

TASK_EXPIRE_SEC = 86400 


def background_modal_inference(task_id: str, temp_path: str):
    """Фоновая задача инференса в Modal"""
    try:
        redis_client.set(f"task:{task_id}:status", "processing")
        
        # Запуск тяжелой аналитики
        analysis_result = run_large_analysis(temp_path)
        
        # Сохраняем успешный результат
        redis_client.set(f"task:{task_id}:result", json.dumps(analysis_result.dict()))
        redis_client.set(f"task:{task_id}:status", "completed")
        
        # Обновляем TTL для успешной задачи
        redis_client.expire(f"task:{task_id}:result", TASK_EXPIRE_SEC)
        redis_client.expire(f"task:{task_id}:status", TASK_EXPIRE_SEC)
        redis_client.expire(f"task:{task_id}:meta", TASK_EXPIRE_SEC)

    except Exception as e:
        # Принудительно пишем traceback ошибки в консоль Docker (stdout)
        print(f"--- КРИТИЧЕСКАЯ ОШИБКА В ФОНОВОМ ПОТОКЕ TASK {task_id} ---")
        traceback.print_exc()
        print("-------------------------------------------------------")
        
        # Записываем ошибку в Redis для фронтенда
        redis_client.set(f"task:{task_id}:status", "failed")
        redis_client.set(f"task:{task_id}:error", str(e))
        redis_client.expire(f"task:{task_id}:status", TASK_EXPIRE_SEC)
        redis_client.expire(f"task:{task_id}:error", TASK_EXPIRE_SEC)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/analyze-video", status_code=status.HTTP_202_ACCEPTED)
async def analyze_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    Принимает файл, сохраняет метаданные и запускает инференс.
    """
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Разрешены только файлы видео")

    task_id = str(uuid.uuid4())
    temp_path = f"temp_{task_id}_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    meta_data = {
        "task_id": task_id,
        "filename": file.filename,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    }
    
    redis_client.set(f"task:{task_id}:status", "pending")
    redis_client.set(f"task:{task_id}:meta", json.dumps(meta_data))
    
    redis_client.expire(f"task:{task_id}:status", TASK_EXPIRE_SEC)
    redis_client.expire(f"task:{task_id}:meta", TASK_EXPIRE_SEC)
    
    background_tasks.add_task(background_modal_inference, task_id, temp_path)

    return {"task_id": task_id, "status": "pending"}


@router.get("/tasks")
async def get_all_tasks():
    """
    Возвращает список всех существующих задач в системе с их статусами и ошибками.
    """
    meta_keys = redis_client.keys("task:*:meta")
    tasks_list = []
    
    for key in meta_keys:
        key_str = key.decode("utf-8")
        task_id = key_str.split(":")[1]
        
        status_bytes = redis_client.get(f"task:{task_id}:status")
        meta_bytes = redis_client.get(f"task:{task_id}:meta")
        
        task_status = status_bytes.decode("utf-8") if status_bytes else "unknown"
        
        if meta_bytes:
            task_meta = json.loads(meta_bytes.decode("utf-8"))
            task_meta["status"] = task_status
            
            # ИСПРАВЛЕНО: Если задача упала, сразу подтягиваем текст ошибки в общий список
            if task_status == "failed":
                error_bytes = redis_client.get(f"task:{task_id}:error")
                task_meta["error"] = error_bytes.decode("utf-8") if error_bytes else "Unknown error"
                
            tasks_list.append(task_meta)
        else:
            tasks_list.append({"task_id": task_id, "status": task_status})
            
    tasks_list.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    
    return {"tasks": tasks_list}


@router.get("/result/{task_id}", response_model=AnalysisResponse)
async def get_analysis_result(task_id: str):
    """
    Возвращает результат конкретной задачи по её task_id.
    """
    status_bytes = redis_client.get(f"task:{task_id}:status")
    if not status_bytes:
        raise HTTPException(status_code=404, detail="Задача не найдена или истёк срок её хранения")

    current_status = status_bytes.decode("utf-8")

    if current_status in ["pending", "processing"]:
        raise HTTPException(
            status_code=202, 
            detail={"status": current_status, "message": "Видео еще обрабатывается в облаке..."}
        )
    elif current_status == "failed":
        error_bytes = redis_client.get(f"task:{task_id}:error")
        error_msg = error_bytes.decode("utf-8") if error_bytes else "Unknown error"
        raise HTTPException(status_code=500, detail=f"Ошибка инференса: {error_msg}")
        
    elif current_status == "completed":
        result_bytes = redis_client.get(f"task:{task_id}:result")
        if not result_bytes:
            raise HTTPException(status_code=500, detail="Данные результата потеряны")
        return json.loads(result_bytes.decode("utf-8"))
