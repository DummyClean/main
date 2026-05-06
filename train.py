import json
import argparse
#필요한 라이브러리가 있다면 import
#반드시 requirements.txt 갱신해두세요

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def setup_train(config):
    # 1. 경로 및 모델 아키텍처 로드
    model_arch = config["model"]["primary_arch"]
    save_dir = config["paths"]["save_dir"]
    data_yaml_path = "./data.yaml"
    
    # 2. 훈련 파라미터 병합 config["training"]
    train_args = {
        "data": data_yaml_path,
        "project": save_dir,
        "name": "",         #YOLO의 'train' 하위 폴더 생성 방지
        "exist_ok": True    #동일한 폴더에 덮어쓰기
    }
    train_args.update(config["training"])

    print(f"--- 베이스라인 훈련 환경 설정 완료 ---")
    print(f"사용 모델: {model_arch}")
    print(f"저장 경로: {save_dir}")
    print(f"적용된 훈련 파라미터: {config['training']}")

    # TODO: 모델링 팀 구현 영역
    #
    # 1. 객체 초기화 (model_arch)
    # 2. 파라미터 연동 (train_args)
    # 3. 결과물은 save_dir에 저장
    # # 폴더 중복 방지 및 덮어쓰기 옵션은 이미 반영됨

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 베이스라인 훈련 스켈레톤")
    parser.add_argument('--config', type=str, default='config.json', help='환경 설정 파일 경로')
    args = parser.parse_args()

    setup_train(load_config(args.config))