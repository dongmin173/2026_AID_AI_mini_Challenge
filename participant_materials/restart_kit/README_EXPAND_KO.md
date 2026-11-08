# 학습 데이터 확대: 2,033 → 2,324장

기존 TTA 비교는 none이 선택되어 최고0.0064와 실질적으로 같은 추론 설정입니다.
이 추가 기능은 기존 train과 select를 합쳐 학습하고, calibration/audit은 유지합니다.
train_expand.py를 기존 restart_kit 폴더에 넣으세요. 기존 EMA Addon이 설치되어 있어야 합니다.

```powershell
cd "C:\Users\ehdal\OneDrive\Desktop\AI_challenge\participant_materials\restart_kit"
$py = (Resolve-Path "..\local_gpu_kit\.venv\Scripts\python.exe").Path
& $py train_expand.py fit --data .. --splits runs\regnet32_fp32\splits.json --arch regnet32 --size 288 --precision fp32 --batch 8 --epochs 5 --lr 5e-6 --head-lr 2.5e-5 --ema-decay 0.99 --init runs\regnet32_ema288\model.pt --out runs\regnet32_expand288
```

현재 최적288 체크포인트에서 낮은 학습률로 5epoch 추가 학습합니다.
이는 기존 optimizer 상태를 재개하는 것이 아니라 새 AdamW로 미세조정하는 것입니다.
원래 splits.json은 그대로 두고, 새 run 안에 다음 파일을 저장합니다.

- original_splits.json: 원래 분할
- splits.json: 이전 select를 train으로 합친 새 분할
- training_config.json: 실제 장수와 모든 학습 옵션

예상 역할은 train2324 / select0 / calibration291 / audit291입니다.
추가 학습에는 제공된 train 폴더의 이미지들만 사용하고 test는 사용하지 않습니다.
온도 보정용 calibration과 audit은 가중치 및 BN 통계 학습에서 제외합니다.

기존 select291장도 이제 학습 자료입니다. 따라서 그 자료로 epoch를 선택하면 평가 누수가 됩니다.
이번 정책은 실행 전에 고정한 **5epoch의 마지막 EMA 가중치**를 제출 후보로 저장하는 것입니다.
학습 중 Select/Calibration/Audit 점수는 계산하지 않습니다. 로그에 Select가 없어도 정상입니다.
일반/EMA 중 무엇이 좋은지 검증으로 고르는 방식도 이번에는 사용하지 않습니다.
학습 종료 후 확장한 학습 이미지들로 EMA의 BN 통계를 한 번 재계산합니다.
best.json은 기존 패키저와 호환하기 위한 파일명이며 select_logloss=null이 정상입니다.
이 파일은 마지막 epoch를 사전에 선택했다는 정책을 기록합니다.

학습 후 별도 calibration291장에서 온도를 새로 구합니다.

```powershell
& $py finalize_ema.py --data .. --run runs\regnet32_expand288 --calibrate --batch 8
```

기존 모델의 T=1.310046을 새 가중치에 그대로 적용하지 않습니다.
보정 후 calibration 손실은 온도를 맞춘 자료 자체의 값이므로 독립 평가 점수는 아닙니다.
Audit은 반복 평가하지 않습니다. 원래 select로 TTA 비교도 하지 않습니다.

CSV/ZIP 생성:

```powershell
& $py make_package.py --run runs\regnet32_expand288 --test-dir ..\test --sample ..\sample_submission.csv --output submission_regnet32_expand288_package.zip --csv submission_regnet32_expand288.csv --include-initial
```

실제 추론에는 model.pt 하나만 사용합니다. initial_model.pt는 오프라인 학습 재현용입니다.
파라미터 수는17,926,364개이며 Colab CPU 시간과 재현성은 새 ZIP으로 확인합니다.
기존0.0064 CSV/ZIP을 보관하세요. 추가 데이터가 Public/Private 점수 개선을 보장하지 않습니다.
학습 자료를 합친 결과를 기존 Select0.008886과 직접 비교해서는 안 됩니다.

이번 파일은 CPU 합성 이미지에서 역할 병합/학습/보정/패키징을 검증합니다.
실제 데이터와 CUDA 학습은 사용자 환경에서 실행해야 합니다.
