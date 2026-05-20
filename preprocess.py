import os
import shutil
import re

def organize_dataset(raw_data_dir='dataset', target_dir='data'):
    """
    원본 클래스별 폴더 구조를 YOLOv8 표준 구조(train/val)로 분류
    """
    classes = ['box', 'coated_paper', 'normal_vinyl', 'clear_vinyl', 
               'disposable_plastic', 'clear_plastic', 'pet', 'styrofoam']
    
    # 대상 디렉토리 생성
    for split in ['train', 'val', 'test']:
        for sub in ['images', 'labels']:
            os.makedirs(os.path.join(target_dir, split, sub), exist_ok=True)

    print(f" Starting data organization from {raw_data_dir} to {target_dir}...")

    for cls in classes:
        cls_path = os.path.join(raw_data_dir, cls)
        if not os.path.exists(cls_path):
            print(f"[!] Folder not found: {cls_path}")
            continue

        files = os.listdir(cls_path)
        for file in files:
            # 파일명에서 숫자 추출 (예: box_106.jpg -> 106)
            match = re.search(r'_(\d+)\.', file)
            if not match:
                continue
            
            num = int(match.group(1))
            if num <= 100:
                target_split = 'train'
            elif num <= 112:
                target_split = 'val'
            else:
                target_split = 'test'
            
            # 확장자에 따라 images/labels 분류
            if file.endswith(('.jpg', '.jpeg', '.png')):
                dest_sub = 'images'
            elif file.endswith('.txt'):
                dest_sub = 'labels'
            else:
                continue

            src_file = os.path.join(cls_path, file)
            dest_file = os.path.join(target_dir, target_split, dest_sub, file)
            
            shutil.copy2(src_file, dest_file)

    print(" Data organization completed.")

if __name__ == "__main__":
    # 사용자가 dataset 폴더를 준비했다고 가정
    if os.path.exists('dataset'):
        organize_dataset()
    else:
        print(" 'dataset' directory not found. Please place your raw folders in 'dataset/'.")
