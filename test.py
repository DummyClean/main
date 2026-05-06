import json
import argparse
import os
import time
#필요한 라이브러리가 있다면 import
#반드시 requirements.txt 갱신해두세요

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def setup_test(config):
    inference_limit = config["model"]["inference_time_limit_sec"]
    save_dir = config["paths"]["save_dir"]
    dataset_root = config["data"]["dataset_root"]
    
    trained_weights = os.path.join(save_dir, "weights", "best.pt")
    
    print(f"--- 테스트 환경 설정 완료 ---")
    print(f"테스트 가중치: {trained_weights}")
    print(f"목표 추론 시간: 장당 {inference_limit}초 이하")

    # TODO: 모델링 팀 구현 영역
    # 1. `trained_weights` 경로에서 학습된 모델을 불러올 것
    # 2. 모델 평가: 검증 데이터셋에 대한 평가 지표(mAP50등)를 콘솔에 출력
    # 3. 추론 시간 측정 로직
    # 4. 장당 평균 추론 시간을 계산
    #    이 값이 config의 inference_limit를 초과할 경우 경고 출력


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    args = parser.parse_args()

    setup_test(load_config(args.config))