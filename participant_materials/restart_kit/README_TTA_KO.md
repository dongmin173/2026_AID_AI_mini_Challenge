# 현재 RegNet 가중치의 TTA + 온도 비교

압축 안의 compare_tta.py를 기존 restart_kit 폴더에 넣습니다.
같은 폴더에 기존 model.py, train.py, data.py, make_package.py가 있어야 합니다.
기존 runs/regnet32_fp32 폴더에는 모델 및 학습 당시 source/ 스냅샷이 필요합니다.

기존 PowerShell에서:

```powershell
& $py compare_tta.py --run runs\regnet32_fp32 --data .. --out runs\regnet32_tta_cal --batch 8
```

새 PowerShell 창에서 시작한 경우:

```powershell
cd "C:\Users\ehdal\OneDrive\Desktop\AI_challenge\participant_materials\restart_kit"
$py = (Resolve-Path "..\local_gpu_kit\.venv\Scripts\python.exe").Path
& $py compare_tta.py --run runs\regnet32_fp32 --data .. --out runs\regnet32_tta_cal --batch 8
```

학습 없이 저장된 model.pt를 평가합니다. 가중치 다운로드도 없습니다.
기존 학습 당시 source/의 모델, 전처리, 평가 함수를 사용합니다.
Select 자료에서 T=1인 원본/좌우 반전 평균을 비교해 추론 방식을 선택합니다.
동점은 추론이 빠른 원본을 선택합니다.
Calibration 자료에서 각 방식의 온도를 0.75~2.0 사이로 최적화합니다.
T=1도 후보에 포함하여 calibration 손실이 개선될 때만 보정을 적용합니다.
최종 후보는 Select로 선택한 추론 방식과 그 방식의 온도를 사용합니다.
TTA에서는 각 방향 logit에 온도를 적용하고 확률을 구한 뒤 평균합니다.
가중치와 BatchNorm 통계는 바꾸지 않습니다.

출력 해석:

- select(T=1): 원본과 TTA를 고르는 기준. 낮을수록 좋습니다.
- cal(T=1): 온도 보정 전 calibration 손실.
- fitted T: 보정 온도. 1보다 작으면 더 확신하고, 크면 확신을 낮춥니다.
- cal(after fit): 온도를 맞춘 자료 자체에서의 손실. 독립 평가 점수는 아닙니다.
- CHOSEN: 실제 새 후보에 저장한 TTA/온도 설정.

결과는 runs/regnet32_tta_cal에 저장합니다. 원래 run은 수정하지 않습니다.
새 폴더가 이미 존재하면 덮어쓰지 않고 중단합니다.
comparison.csv와 postprocess_comparison.json에 수치와 선택 규칙을 기록합니다.
training_config.json과 best.json은 원래 학습 기록입니다. 새 추론 설정은 inference_config.json입니다.

기본 명령은 audit 예측을 하지 않습니다. audit_metrics.json에는 미평가 상태를 명시합니다.
설정을 고정한 뒤 Audit을 1회 확인하기로 미리 결정했다면 최초 비교 명령에 --audit을 추가합니다.
Audit 결과로 원본/TTA/온도를 다시 고르지 마세요. 테스트 이미지는 비교/보정에 전혀 사용하지 않습니다.

새 후보는 이미 설정이 고정되어 있으므로 finalize.py를 다시 실행하지 않습니다.
CSV와 최종 ZIP 생성:

```powershell
& $py make_package.py --run runs\regnet32_tta_cal --test-dir ..\test --sample ..\sample_submission.csv --output submission_regnet32_tta_cal_package.zip --csv submission_regnet32_tta_cal.csv
```

기존 make_package.py가 새 후보의 설정을 사용해 ZIP을 만들고, ZIP을 새 폴더에 풀어
CPU에서 submission_regnet32_tta_cal.csv를 생성합니다. 비교 코드와 선택 기록도 ZIP에 포함됩니다.
기존 0.0092 CSV/ZIP은 그대로 보관하세요. 최종 선택 CSV와 업로드 ZIP의 짝을 맞추세요.
새 ZIP의 Colab CPU 시간/CSV 재현 확인은 이전과 같은 절차로 수행합니다.
보정으로 Public 점수가 좋아진다고 보장할 수는 없습니다.

검증: torch 2.10 CPU에서 합성 이미지와 실제 RegNet 구조로 비교/보정/후보 생성,
기존 패키저를 통한 ZIP 추출 및 500장 CSV 생성을 시험했습니다.
실제 대회 가중치/이미지의 성능과 RTX 3070 Ti 실행은 사용자 환경에서 확인해야 합니다.
