# RegNetY 224 + EMA

train_ema.py와 finalize_ema.py를 기존 restart_kit 폴더에 넣으세요.
기존 모델/학습/전처리/패키징 파일들이 있는 폴더에서 실행합니다.

```powershell
cd "C:\Users\ehdal\OneDrive\Desktop\AI_challenge\participant_materials\restart_kit"
$py = (Resolve-Path "..\local_gpu_kit\.venv\Scripts\python.exe").Path
& $py train_ema.py fit --data .. --splits runs\regnet32_fp32\splits.json --arch regnet32 --size 224 --precision fp32 --batch 8 --epochs 20 --ema-decay 0.99 --init runs\regnet32_fp32\initial_model.pt --out runs\regnet32_ema224
```

이전 실험의 initial_model.pt에서 새로운 학습을 시작합니다.
※ 위 초기 파일은 이전 실험 시작 시 저장한 ImageNet 백본 + 초기 출력층입니다.
이번 실험은 같은 데이터 분할, 초기 가중치, 증강, 학습률, 시드에서 EMA를 추가합니다.
입력224, FP32, batch8, 20epoch, cosine 감소, TTA none, 온도1로 모델을 선택합니다.

처음 1epoch는 기존과 같은 head warmup이며 EMA를 누적하지 않습니다.
이후 성공한 optimizer step마다 EMA decay0.99로 평균을 갱신합니다.
각 epoch에서 EMA의 BatchNorm 통계를 학습 부분 이미지로 재계산하고 Select를 평가합니다.
BN 재계산에는 eval 전처리와 고정 난수로 섞은 별도 DataLoader를 사용합니다. calibration/audit/test는 사용하지 않습니다.
일반 모델(raw)과 EMA의 모든 epoch 중 Select Log Loss가 가장 낮은 상태를 model.pt로 저장합니다.
EMA가 더 좋지 않으면 raw 모델이 선택될 수 있습니다. EMA를 무조건 제출하도록 강제하지 않습니다.
BN 통계 재계산과 EMA 검증 때문에 기존 학습보다 시간이 추가됩니다.

출력:

- raw: 일반 모델의 현재 epoch Select 손실
- EMA: EMA 모델의 Select 손실, 첫 warmup epoch는 warmup 표시
- best=점수(raw/ema): 현재까지 가장 좋은 모델의 점수와 종류
- best.json: 실제 선택 epoch, variant, select_logloss
- selection_summary.json: raw/EMA 각각의 최적 결과
- raw_best.pt/ema_best.pt: 비교용 가중치
- model.pt: 실제 제출할 단 하나의 모델

학습이 끝나면 선택된 model.pt의 온도를 별도 calibration 291장에서 학습합니다.
기존 모델에서 구한 T=0.931048을 새 모델에 그대로 적용하지 않습니다.

```powershell
& $py finalize_ema.py --data .. --run runs\regnet32_ema224 --calibrate --batch 8
```

설정을 고정하고 inference_config.json을 만듭니다. Audit은 반복 평가하지 않습니다.
calibration의 보정 후 손실은 온도를 맞춘 자료의 점수이며 독립 성능 검증은 아닙니다.
이후 같은 폴더에서 다음 명령으로 새 CSV와 ZIP을 만듭니다.

```powershell
& $py make_package.py --run runs\regnet32_ema224 --test-dir ..\test --sample ..\sample_submission.csv --output submission_regnet32_ema224_package.zip --csv submission_regnet32_ema224.csv
```

패키저는 model.pt만 포함합니다. raw_best.pt와 ema_best.pt를 둘 다 추론하는 앙상블이 아닙니다.
파라미터는17,926,364개이며 입력해상도/TTA도 동일하므로 추론 비용은 기존과 유사한 구조입니다.
실제 Colab CPU 시간과 CSV 재현은 새 ZIP/CSV로 다시 확인하세요.
학습 및 비교 코드, 선택 기록도 패키지에 포함됩니다.

학습 전 기존 out 폴더가 있으면 새 이름을 쓰세요. 중단된 학습의 재개 기능은 없습니다.
메모리가 부족하면 새 out 폴더에서 batch4, bn-batch8로 시작하세요.
기존0.0085 CSV/ZIP은 보관하고, 새 점수 확인 후 최종 평가 CSV와 해당 ZIP을 맞추세요.

검증은 CPU/합성 이미지/64px에서 실제 RegNet 구조로 수행합니다.
실제 대회 데이터의 Log Loss 개선과 CUDA 실행은 사용자 환경에서 확인해야 합니다.
