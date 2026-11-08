/restart_kit
# AI Mini Challenge 제출 모델

## 모델 및 학습 방식

- **모델:** ImageNet 사전학습 RegNetY-3.2GF, 출력층을 2개 클래스로 변경
- **파라미터 수:** 17,926,364개
- **최종 입력 크기:** 288 × 288
- **클래스:** 제6공학관 `0`, IT관 `1`
- **학습 방식:** 224 해상도 미세조정 및 EMA 적용 → 288 해상도 미세조정 → 학습 자료를 2,324장으로 확대하여 5 epoch 추가 학습
- **최종 가중치:** 확대 학습의 마지막 EMA 가중치
- **확률 보정:** Temperature scaling, `T = 0.935492`
- **TTA:** 사용하지 않음

학습에는 AdamW, cosine 학습률 감소, 좌우 반전·약한 crop·색상 증강을 사용했습니다. 최종 확대 학습에서는 Calibration 291장과 Audit 291장을 가중치 및 BatchNorm 통계 학습에서 제외했습니다. 테스트 이미지는 학습과 확률 보정에 사용하지 않았습니다.

## 추론에 필요한 파일

압축을 해제한 뒤 아래 파일들을 같은 폴더에 두면 됩니다.

| 파일 | 역할 |
|---|---|
| `predict.py` | CPU 추론 및 제출 CSV 생성 |
| `model.py` | 모델 구조와 이미지 전처리 |
| `model.pt` | 최종 모델의 전체 가중치 |
| `inference_config.json` | 입력 크기, 온도, 클래스 매핑 등 추론 설정 |
| `requirements_cpu.txt` | 실행 환경의 패키지 및 버전 |

`initial_model.pt`는 마지막 추가 학습의 시작 가중치이며, 학습 재현용으로 포함했습니다. **추론에는 `model.pt` 하나만 사용합니다.**

## 실행 방법

### 1. 환경 준비

필요한 패키지가 설치되어 있지 않거나 버전이 다르면, 추론 전에 다음 명령으로 환경을 준비합니다.

```bash
python -m pip install -r requirements_cpu.txt
```

### 2. CPU 추론

압축을 해제한 폴더에서 실행합니다. `--test_dir`에는 테스트 이미지 500장이 들어 있는 폴더를 지정합니다.

```bash
python predict.py --test_dir /content/test --output /content/submission.csv
```

모델은 CPU에서 실행되며, 추론 중 사전학습 가중치나 외부 파일을 다운로드하지 않습니다. 이미지 로딩·전처리·모델 로딩·추론·CSV 저장을 포함한 실행 시간이 출력됩니다.

## 출력 형식

```csv
id,prob
이미지파일명에서_확장자를_제외한_ID,IT관일_확률
```

- 테스트 이미지 500장에 대해 헤더를 제외한 500개 행을 생성합니다.
- `prob`는 IT관, 즉 클래스 `1`일 확률입니다.
- 이미지별 확률은 `inference_config.json`의 온도로 보정합니다.

## 학습 기록

- `train_expand.py`: 최종 확대 학습 코드
- `finalize_ema.py`: 온도 보정 및 추론 설정 저장 코드
- `training_config.json`: 최종 추가 학습의 실행 옵션
- `splits.json`: 최종 학습·Calibration·Audit 분할과 이미지 체크섬
- `original_splits.json`: 확대 전 데이터 분할

운영진의 예측 재실행에는 재학습 없이 **위 CPU 추론 명령만 실행**하면 됩니다.

깃허브 링크 
