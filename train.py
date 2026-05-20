import json
import argparse
import os
import shutil
import time
from ultralytics import YOLO
from preprocess import organize_dataset
from pathlib import Path


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def setup_train(config):
    # 1. 경로 및 모델 아키텍처 로드
    current_dir = Path(__file__).resolve().parent
    model_arch     = config["model"]["default_model"]
    save_dir       = config["paths"]["save_dir"]
    data_yaml_path = current_dir / "data.yaml"

    # 2. 훈련 파라미터 병합 config["training"]
    train_args = {
        "data"    : data_yaml_path,
        "project" : save_dir,
        "name"    : "",       # YOLO의 'train' 하위 폴더 생성 방지
        "exist_ok": True      # 동일한 폴더에 덮어쓰기
    }
    train_args.update(config["training"])

    print(f"--- 베이스라인 훈련 환경 설정 완료 ---")
    print(f"사용 모델: {model_arch}")
    print(f"저장 경로: {save_dir}")
    print(f"적용된 훈련 파라미터: {config['training']}")

    # ──────────────────────────────────────────────────────────────
    # TODO: 모델링 팀 구현 영역
    # ──────────────────────────────────────────────────────────────

    # 데이터 전처리 — dataset/ 폴더가 있고 data/train/images 가 없을 때만 실행
    raw_dir       = config.get("data", {}).get("raw_dir", "dataset")
    processed_dir = config.get("data", {}).get("processed_dir", "data")
    if os.path.exists(raw_dir) and \
       not os.path.exists(os.path.join(processed_dir, "train", "images")):
        print("[*] Raw dataset 감지 — YOLOv8 형식으로 구성")
        organize_dataset(raw_dir, processed_dir)

    # 1. 객체 초기화 (default_model = yolov8s.pt)
    model = _train_model(model_arch, train_args)

    # ── 추론 속도 체크 ──────────────────────────────────────────
    val_img = _get_sample_image(config["data"]["val_path"])

    if val_img:
        best_path = os.path.join(save_dir, "best.pt")
        if not os.path.exists(best_path):
            print("best.pt 찾을 수 없음")
            return
        
        inf_time = _measure_inference(os.path.join(save_dir, "best.pt"), val_img, config["model"])

        if inf_time > config["model"]["inference_time_limit_sec"]:
            print(f"추론 시간 {inf_time:.4f}s > 기준 {config['model']['inference_time_limit_sec']}s")
            print(f"Fallback 모델({config['model']['fallback_model']})로 재훈련을 시작합니다.")
            fallback_args         = dict(train_args)
            fallback_args["name"] = "fallback"
            model = _train_model(config["model"]["fallback_model"], fallback_args)
            _copy_best(model.trainer.save_dir, save_dir)
        else:
            print(f"추론 시간 {inf_time:.4f}s ≤ 기준 {config['model']['inference_time_limit_sec']}s — {model_arch} 유지")
    else:
        print("val 이미지를 찾을 수 없어 속도 체크를 건너뜁니다.")

    # ──────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────
# 내부 헬퍼 함수
# ─────────────────────────────────────────────

def _train_model(arch: str, train_args: dict) -> YOLO:
    """주어진 아키텍처로 훈련하고 YOLO 객체를 반환"""
    print(f"[*] Initializing YOLO model: {arch}")
    model = YOLO(arch)

    history = {"epoch": [], "map50": [], "map5095": [], "loss": []}

    def _log_epoch(trainer):
        ep = trainer.epoch + 1
        m  = trainer.metrics
        tl = trainer.loss_items
        history["epoch"]   .append(ep)
        history["map50"]   .append(m.get("metrics/mAP50(B)",    0))
        history["map5095"] .append(m.get("metrics/mAP50-95(B)", 0))
        history["loss"]    .append(float(tl.mean()) if tl is not None else 0)
        print(f"  [Epoch {ep:>3}] loss={history['loss'][-1]:.4f} "
              f"mAP50={history['map50'][-1]:.4f} "
              f"mAP50-95={history['map5095'][-1]:.4f}")

    model.add_callback("on_fit_epoch_end", _log_epoch)

    print(f"Training started: {arch}")
    start   = time.time()
    results = model.train(**train_args)
    elapsed = time.time() - start
    print(f"Training completed in {elapsed / 60:.1f} min")

    _copy_best(str(results.save_dir), train_args["project"])

    return model


def _get_sample_image(val_path: str) -> str | None:
    """val 폴더에서 이미지 한 장의 경로를 반환"""
    if not os.path.exists(val_path):
        return None
    imgs = [f for f in os.listdir(val_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return os.path.join(val_path, imgs[0]) if imgs else None


def _measure_inference(model_path: str, img_path: str, model_cfg: dict) -> float:
    """best.pt 로 이미지 한 장 추론 후 소요 시간(초) 반환"""
    print(f"추론 속도 측정 중: {model_path}")
    m     = YOLO(model_path)

    #첫 사진 오버헤드 방지용 워밍업
    m.predict(img_path, imgsz=model_cfg["imgsz"], verbose=False)

    results = m.predict(img_path, imgsz=model_cfg["imgsz"],
                       conf=model_cfg["conf_threshold"], verbose=False)
    
    inference_time = results[0].speed['inference'] / 1000.0
    return inference_time


def _copy_best(run_dir: str, save_dir: str): # 결과물 저장
    """훈련 결과 폴더의 best.pt 를 save_dir 루트에 복사""" 
    src = os.path.join(run_dir, "weights", "best.pt")
    dst = os.path.join(save_dir, "best.pt")
    if os.path.exists(src):
        os.makedirs(save_dir, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Best weights → {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 베이스라인 훈련 스켈레톤")
    parser.add_argument('--config', type=str, default='config.json', help='환경 설정 파일 경로')
    args = parser.parse_args()
    
    setup_train(load_config(args.config))
