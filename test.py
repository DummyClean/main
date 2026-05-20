import argparse
import json
import os
import warnings
import torch
from ultralytics import YOLO
from pathlib import Path


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def setup_test(config):
    try:
        current_dir = Path(__file__).resolve().parent
    except NameError:
        current_dir = Path.cwd()

    inference_limit = config["model"]["inference_time_limit_sec"]
    save_dir = config["paths"]["save_dir"]
    trained_weights = os.path.join(save_dir, "best.pt")

    print(f"--- 테스트 환경 설정 완료 ---")
    print(f"테스트 가중치: {trained_weights}")
    print(f"목표 추론 시간: 장당 {inference_limit}초 이하")

    if not os.path.exists(trained_weights):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {trained_weights}")

    print(f"\n[1/3] 모델 로딩 중: {trained_weights}")
    model = YOLO(trained_weights)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f" 디바이스: {device}")

    print(f"\n[2/3] GPU 워밍업 중 ")
    warmup_source = config["data"].get("val_img_dir", config["data"]["val_path"])

    if warmup_source:
        warmup_path = Path(warmup_source)

        if not warmup_path.exists():
            print(f" 워밍업 경로를 찾을 수 없어 건너뜁니다: {warmup_path}")
        else:
            valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

            if warmup_path.is_file() and warmup_path.suffix.lower() in valid_extensions:
                warmup_images = [str(warmup_path)]
            elif warmup_path.is_file():
                warmup_images = []
            else:
                warmup_images = [
                    str(p) for p in warmup_path.rglob("*")
                    if p.is_file() and p.suffix.lower() in valid_extensions
                ]

            for img in warmup_images[:3]:
                model.predict(img, verbose=False)

            if device == "cuda":
                torch.cuda.synchronize()

            print(f" 워밍업 완료: {min(3, len(warmup_images))}장")
    else:
        print(" 워밍업 이미지 경로가 없어 건너뜁니다.")

    test_data = config["data"].get("val_yaml", os.path.join(current_dir, "data.yaml"))
    
    print(f"\n[3/3] 검증 데이터셋 평가 및 속도 측정 중: {test_data}")

    results = model.val(data=test_data, split = 'test' ,verbose=True)

    metrics = results.results_dict
    print("\n === 평가 지표 ===")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    avg_preprocess = results.speed.get("preprocess", 0) / 1000.0
    avg_inference = results.speed.get("inference", 0) / 1000.0
    avg_postprocess = results.speed.get("postprocess", 0) / 1000.0
    avg_total_time = avg_preprocess + avg_inference + avg_postprocess

    print(f"\n === 추론 시간 결과 (평균, 이미지 1장 기준) ===")
    print(f"  전처리: {avg_preprocess:.4f}초")
    print(f"  순수 추론: {avg_inference:.4f}초")
    print(f"  후처리: {avg_postprocess:.4f}초")
    print(f"  총 소요: {avg_total_time:.4f}초")

    check_time = avg_inference  # 또는 avg_total_time

    print(f"\n 제한 시간 검사: {check_time:.4f}초 (한계: {inference_limit}초)")
    if check_time > inference_limit:
        warnings.warn(
            f"평균 추론 시간({check_time:.4f}초)이 설정된 한계({inference_limit}초)를 초과합니다!",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        print("추론 시간 기준 통과")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    args = parser.parse_args()
    
    setup_test(load_config(args.config))
