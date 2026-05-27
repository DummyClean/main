# 더미클린 (Dummy Clean)

YOLOv8을 활용한 8종 객체 탐지 및 분류 파이프라인입니다.

## 프로젝트 일정 및 개발 목표
- **~ 5/13** : 모델 파이프라인 구축 완료 및 데이터 라벨링
- **~ 5/20** : 모델 구조 및 정확도 개선
- **~ 5/27** : 하이퍼파라미터 튜닝 및 외부 I/O 구축 1차
- **~ 6/03** : 하이퍼파라미터 튜닝 및 외부 I/O 구축 2차
- **~ 6/10** : 최종 발표 준비 및 유예기간

## 모델 정보 (Modeling)
- **기본 모델** : YOLOv8 Small
- **대체 모델** : YOLOv8 Nano (이미지 1장 판별 시 0.1초 이상 소요될 경우 속도 개선을 위해 자동 변경)
- **환경 설정** : `config.json` 파일을 통해 모델 파라미터 및 경로 등을 통합 관리합니다.

## 데이터셋 (Dataset)
`labelimg` 툴을 이용하여 라벨당 125장씩 라벨링을 진행하였습니다.
데이터는 YOLO 포맷을 따르며, `train`, `test`, `val` 데이터셋으로 구분되어 관리됩니다.

데이터셋 다운로드 링크
https://drive.google.com/file/d/1l3bJSyOZgWrM8TmZKb0LyUbEnL126-L_/view?usp=sharing

### 클래스 구성 (8 Classes)
1. `box` (박스)
2. `coated_paper` (코팅지)
3. `normal_vinyl` (일반 비닐봉지)
4. `clear_vinyl` (투명 비닐/필름)
5. `disposable_plastic` (일회용 플라스틱 용기)
6. `clear_plastic` (투명 플라스틱 용기)
7. `pet` (페트병)
8. `styrofoam` (스티로폼)

## 환경 설정 및 요구사항 설치
프로젝트에 필요한 모든 패키지 및 라이브러리는 `requirements.txt`에 명시되어 있습니다.
```bash
pip install -r requirements.txt
```
```open_cv_test.py``` 는 본인 휴대폰에 IP Wepcam 앱 실행 후 IP 입력을 해줘야합니다

## 파라미터 튜닝 기록
https://docs.google.com/spreadsheets/d/1v333uxaEcliwWmrqhNiyWYiNTAY7rs_BWYICTLtC6vM/edit?usp=sharing
