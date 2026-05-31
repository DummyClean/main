import os
import shutil
import re
import random

def organize_dataset_scalable(raw_data_dir='dataset', target_dir='data'):
    # ==========================================
    # 튜닝 파라미터 (확장성 고려)
    # ==========================================
    train_ratio = 0.8  # 학습 데이터 비율 (80%)
    val_ratio = 0.1    # 검증 데이터 비율 (10%)
    group_size = 5     # 동일 객체를 촬영한 연속 사진 수 (묶음 단위)
    seed = 42          # 재현성을 위한 랜덤 시드
    
    random.seed(seed)
    
    classes = ['box', 'coated_paper', 'normal_vinyl', 'clear_vinyl', 
               'disposable_plastic', 'clear_plastic', 'pet', 'styrofoam', 'background']
    
    # 대상 디렉토리 초기화 및 생성
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir) # 기존 데이터 초기화 (찌꺼기 방지)
    
    for split in ['train', 'val', 'test']:
        for sub in ['images', 'labels']:
            os.makedirs(os.path.join(target_dir, split, sub), exist_ok=True)

    print(f"[*] Starting scalable data organization from '{raw_data_dir}' to '{target_dir}'...")
    print(f"[*] Rules -> Split: {train_ratio*100}% / {val_ratio*100}% / {(1-train_ratio-val_ratio)*100}%, Group Size: {group_size}, Seed: {seed}\n")

    for cls in classes:
        cls_path = os.path.join(raw_data_dir, cls)
        if not os.path.exists(cls_path):
            continue

        # 1. 이미지 파일 수집 및 번호순 정렬 (그룹화를 위해 필수)
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        file_data = []
        for file in files:
            match = re.search(r'_(\d+)\.', file)
            if match:
                file_data.append((int(match.group(1)), file))
        
        file_data.sort(key=lambda x: x[0]) # 파일명 내 숫자를 기준으로 오름차순 정렬

        # 2. 지정된 사이즈(group_size)만큼 파일 묶기
        groups = []
        for i in range(0, len(file_data), group_size):
            groups.append(file_data[i:i + group_size])

        # 3. 그룹 단위로 랜덤 셔플 (객관성 확보 및 데이터 누수 방지)
        random.shuffle(groups)

        # 4. 비율에 따른 인덱스 계산
        num_groups = len(groups)
        train_end_idx = int(num_groups * train_ratio)
        val_end_idx = train_end_idx + int(num_groups * val_ratio)

        train_groups = groups[:train_end_idx]
        val_groups = groups[train_end_idx:val_end_idx]
        test_groups = groups[val_end_idx:]

        # 5. 파일 복사 및 라벨 이동 헬퍼 함수
        def process_and_move(groups_list, target_split):
            count = 0
            for group in groups_list:
                for num, file in group:
                    # 이미지 이동
                    src_img = os.path.join(cls_path, file)
                    dest_img = os.path.join(target_dir, target_split, 'images', file)
                    shutil.copy2(src_img, dest_img)

                    # 배경 클래스: 빈 txt 파일 자동 생성
                    if cls == 'background':
                        txt_filename = os.path.splitext(file)[0] + '.txt'
                        txt_dest = os.path.join(target_dir, target_split, 'labels', txt_filename)
                        with open(txt_dest, 'w') as f:
                            pass 
                    # 일반 객체 클래스: 기존 txt 파일 복사
                    else:
                        txt_filename = os.path.splitext(file)[0] + '.txt'
                        src_txt = os.path.join(cls_path, txt_filename)
                        dest_txt = os.path.join(target_dir, target_split, 'labels', txt_filename)
                        if os.path.exists(src_txt):
                            shutil.copy2(src_txt, dest_txt)
                    count += 1
            return count

        # 각 폴더로 분배 실행
        train_cnt = process_and_move(train_groups, 'train')
        val_cnt = process_and_move(val_groups, 'val')
        test_cnt = process_and_move(test_groups, 'test')
        
        print(f"  - [{cls}] Total {len(file_data)} imgs -> Train: {train_cnt} | Val: {val_cnt} | Test: {test_cnt}")

    print("\n[*] Scalable data organization completed successfully.")

if __name__ == "__main__":
    if os.path.exists('dataset'):
        organize_dataset_scalable()
    else:
        print("[!] 'dataset' directory not found.")