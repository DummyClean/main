import argparse
import json
import os
import time
import warnings
import torch
from ultralytics import YOLO
from pathlib import Path


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def setup_test(config):
    current_dir = Path(__file__).resolve().parent
    inference_limit = config["model"]["inference_time_limit_sec"]
    save_dir = config["paths"]["save_dir"]
    trained_weights = os.path.join(save_dir, "best.pt")

    print(f"--- 테스트 환경 설정 완료 ---")
    print(f"테스트 가중치: {trained_weights}")
    print(f"목표 추론 시간: 장당 {inference_limit}초 이하")

    # 1. trained_weights 경로에서 학습된 모델 불러오기
    if not os.path.exists(trained_weights):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {trained_weights}")

    print(f"\n[1/4] 모델 로딩 중: {trained_weights}")
    model = YOLO(trained_weights)          # ← framework's loader로 전환
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f" 디바이스: {device}")

    # 2. 모델 평가: 검증 데이터셋에 대한 평가 지표(mAP50등)를 콘솔에 출력
    val_data = config["data"].get("val_yaml", os.path.join(current_dir, "data.yaml"))
    print(f"\n[2/4] 검증 데이터셋 평가 중: {val_data}")
    results = model.val(data=val_data, verbose=True)

    metrics = results.results_dict          # dict with mAP50, mAP50-95, etc.
    print("\n  === 평가 지표 ===")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # 3 & 4. 추론 시간 측정 로직과 장당 평균 추론 시간을 계산
    print(f"\n[3/4] 추론 시간 측정 중 ...")
    val_img_dir = config["data"].get(
        "val_img_dir", config["data"]["val_path"]
    )

    # val_img_dir가 문자열이라면 Path 객체로 변환합니다.
    val_path = Path(val_img_dir)
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    image_paths = [
        str(p) for p in val_path.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_extensions
    ]

    if not image_paths:
        print(" 경고: 추론할 이미지를 찾지 못했습니다.")
        return

    # Warm-up pass (GPU JIT / cuDNN auto-tuning 제외)
    WARMUP = min(3, len(image_paths))
    print(f"  워밍업 {WARMUP}장 …")
    for img in image_paths[:WARMUP]:
        model.predict(img, verbose=False)

    # Timed pass
    elapsed_times = []
    for img in image_paths:
        t0 = time.perf_counter()
        model.predict(img, verbose=False)
        elapsed_times.append(time.perf_counter() - t0)

    avg_time = sum(elapsed_times) / len(elapsed_times)
    max_time = max(elapsed_times)
    min_time = min(elapsed_times)

    print(f"\n [4/4] === 추론 시간 결과 ({len(elapsed_times)}장) ===")
    print(f"  평균: {avg_time:.4f}초")
    print(f"  최소: {min_time:.4f}초")
    print(f"  최대: {max_time:.4f}초")

    # 이 값이 config의 inference_limit를 초과할 경우 경고 출력
    print(f"\n 제한 시간 검사: {avg_time:.4f}초 (한계: {inference_limit}초)")
    if avg_time > inference_limit:
        warnings.warn(
            f"평균 추론 시간({avg_time:.4f}초)이 "
            f"설정된 한계({inference_limit}초)를 초과합니다!",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        print(f"추론 시간 기준 통과")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    args = parser.parse_args()
    
    setup_test(load_config(args.config))
