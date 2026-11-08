# AI Mini Challenge — 새 실험 시작 키트

기존 B0/B2 파일을 건드리지 않고, 같은 Windows 가상환경에서 새 학습을 시작합니다.
기본 모델은 torchvision RegNetY-3.2GF, 입력 224, ImageNet V2 사전학습입니다.
기존 대회 체크포인트를 이어 학습하지 않습니다. 성능 개선이나 Public 0.02대는 보장하지 않습니다.
이 ZIP은 **코드 키트**이며 학습된 모델이 들어 있는 최종 제출 ZIP은 아닙니다.

## 1. 압축 해제 및 기존 Python 사용

첨부 폴더 목록의 데이터 루트:
`C:\Users\ehdal\OneDrive\Desktop\AI_challenge\participant_materials`

이 폴더 바로 아래에 `restart_kit`이 생기도록 압축을 풉니다.
동일한 루트에 `train\6e`, `train\it`, `test`, `sample_submission.csv`가 있어야 합니다.
첨부 목록에는 `participant_materials`가 한 번 더 중첩된 사본도 있습니다.
이번에는 바깥쪽 데이터 루트를 일관되게 사용하세요.

PowerShell:

```powershell
cd "C:\Users\ehdal\OneDrive\Desktop\AI_challenge\participant_materials"
$py = (Resolve-Path ".\local_gpu_kit\.venv\Scripts\python.exe").Path
cd .\restart_kit
& $py -m pip install -r requirements.txt
& $py check_env.py
```

첨부 목록상 기존 가상환경은 `local_gpu_kit\.venv`입니다.
실제로 `venv`나 `.venv`가 다른 위치에 있으면 `$py` 지정 줄만 해당 경로로 바꾸세요.
새 가상환경 생성, 기존 환경 삭제, torch/torchvision 일괄 업그레이드는 필요 없습니다.
가상환경 활성화도 필요하지 않습니다. `& $py`는 지정한 Python을 직접 실행합니다.
새 PowerShell 창을 열었다면 `$py`를 다시 지정해야 합니다.

`CUDA available: True`, `GPU: NVIDIA GeForce RTX 3070 Ti`, `GPU forward pass OK` 확인 후 진행합니다.
ImageNet 가중치는 **첫 학습 때만** 인터넷으로 내려받습니다. 다운로드 오류를 CPU 문제와 혼동하지 마세요.

## 2. 고정 분할 만들기 — 한 번만 실행

```powershell
& $py train.py prepare --data .. --out splits.json
```

제공된 `train` 2,906장만 읽습니다. `test`는 분할, 학습, 확률 보정에 쓰지 않습니다.
클래스 매핑은 `6e=0`, `it=1`로 고정합니다.

| 부분 | 대략적 비율 | 용도 |
|---|---:|---|
| train | 70% | 가중치 학습 |
| select | 10% | 최적 epoch 선택 |
| calibration | 10% | 선택적으로 temperature 학습 |
| audit | 10% | 모든 선택을 고정한 뒤 마지막 평가 |

같은 픽셀을 가진 이미지들은 같은 부분에 넣습니다. 정확한 장수는 그룹 크기에 따라 달라집니다.
`splits.json`에 파일명, 라벨, 이미지 해시, 역할이 저장됩니다. 학습할 때 데이터 변경 여부를 검사합니다.
`splits_near_duplicates.csv`에는 dHash가 가까운 이미지 쌍이 저장됩니다.
이는 **검토 후보**입니다. 비슷한 건물 사진을 곧바로 중복으로 판단하거나 자동 삭제하지 않습니다.
촬영 시점/장소별 그룹 정보는 없으므로 비슷한 구도의 사진이 서로 다른 부분에 남을 수 있습니다.
중복 후보가 많은 경우 학습 전 눈으로 확인하고, 연속 촬영 그룹을 알 수 있다면 그룹 분할로 개선하세요.

약 290장인 audit에도 표본 오차가 큽니다. audit 하나가 Private 점수의 정확한 예측치는 아닙니다.
원래 검증 약 582장과 이 분할은 다르므로, 새 점수와 과거 0.02760을 직접 비교하지 마세요.
기존 B0/B2는 새 audit의 일부를 학습했을 수 있습니다. 기존 모델을 새 audit에서 평가해
새 모델과 공정하게 비교했다고 해석하면 안 됩니다.

## 3. RegNetY를 새로 학습

```powershell
& $py train.py fit --data .. --splits splits.json --arch regnet32 --size 224 --epochs 20 --batch 8 --workers 0 --precision fp32 --out runs\regnet32_224
```

설정은 다음과 같습니다.

- ImageNet V2 가중치 + 새 2클래스 출력층
- 첫 1 epoch는 출력층만 학습, 이후 전체 모델 미세조정
- AdamW, backbone LR 5e-5, head LR 5e-4, weight decay 1e-4, cosine 감소
- 기본 CUDA FP32, gradient clipping. `--precision fp16`은 선택 사항입니다.
- 작은 crop, 좌우 반전, 약한 색 증강
- 기본은 TTA 없음, label smoothing 없음, MixUp/CutMix 없음
- 가장 낮은 **select Log Loss** epoch의 전체 가중치를 `model.pt`에 저장
- `initial_model.pt`에 해당 실행의 시작 가중치 저장

3070 Ti에서는 FP32 배치 8부터 시작합니다. 메모리 부족 오류가 나면 **새 out 폴더로** `--batch 4`를 사용하세요.
이 키트는 GPU 학습 재개 기능을 제공하지 않습니다. 중단한 run을 덮어쓰지 않고 새 이름으로 시작합니다.
Windows 실행 문제를 줄이기 위해 `workers=0`이 기본입니다. 속도가 충분하면 그대로 두세요.
20 epoch의 실제 소요 시간은 사용자 환경에서 측정해야 합니다.

이전 버전에서 `clip_grad_norm_ ... non-finite` 오류로 중단되었다면, 수정된 `train.py`로
교체하고 아래 명령을 실행하세요. 이전 실행에서 저장한 ImageNet 시작 가중치를 그대로 사용합니다.

```powershell
& $py train.py fit --data .. --precision fp32 --batch 8 --init runs\regnet32_224\initial_model.pt --out runs\regnet32_fp32
```

FP32에서도 GPU는 계속 사용합니다. 기존 `splits.json`을 재사용하고 새 출력 폴더에 처음부터 학습합니다.
이 경우 이후 finalize/package 명령의 `--run`도 `runs\regnet32_fp32`로 바꾸세요.
FP32의 non-finite 검사 자체는 유지합니다. 선택적으로 켠 FP16에서 비정상 gradient가 발생하면
해당 optimizer 업데이트를 건너뛰고 scale을 낮춥니다. 건너뛴 횟수는 로그에 기록되며,
오류가 반복되면 FP32로 전환하라는 메시지와 함께 중단합니다.

학습 로그의 `select`는 checkpoint 선택에 사용한 값입니다. 최종 일반화 성능으로 해석하지 마세요.
같은 select에 수십 가지 설정을 반복 적용하면 여기도 과적합됩니다.

기본 2클래스 모델의 파라미터 수:

| --arch | 모델 | 파라미터 |
|---|---|---:|
| regnet32 | RegNetY-3.2GF | 17,926,364 |
| regnet16 | RegNetY-1.6GF | 10,315,208 |
| b0 | EfficientNet-B0 | 4,010,110 |

3.2GF가 CPU 시간 제한에 맞지 않으면 먼저 더 가벼운 `regnet16`을 검토합니다.
서로 다른 모델을 동시에 사용하지 않습니다. 앙상블 기능은 이 키트에 없습니다.
별도 아키텍처 비교를 한다면 새 모델 모두 같은 `splits.json`과 TTA 설정으로 학습하세요.

## 4. 설정 고정 후 audit 평가

첫 실험 권장 명령은 보정을 추가하지 않는 아래 명령입니다.

```powershell
& $py finalize.py --data .. --run runs\regnet32_224
```

T=1, 학습 때 정한 TTA 설정을 유지한 채 `inference_config.json`을 먼저 고정합니다.
이후에만 audit을 평가해 `audit_metrics.json`과 `audit_predictions.csv`를 만듭니다.
`audit_predictions.csv`는 이미지별 손실이 큰 순서입니다.

확률 보정까지 하기로 **audit을 보기 전에 결정했다면**, 위 명령 대신 다음 명령을 한 번 실행합니다.

```powershell
& $py finalize.py --data .. --run runs\regnet32_224 --calibrate
```

별도 calibration 부분에서만 T를 0.75~2.0 범위로 학습합니다.
TTA가 켜진 실험이라면 실제 TTA 확률 평균을 사용해 T를 최적화합니다.
표본이 작으므로 calibration 손실 개선이 audit/Public 개선을 보장하지 않습니다.
이미 finalize한 run에는 재실행을 막습니다. audit 점수를 본 뒤 보정/TTA를 바꾸어
같은 audit을 계속 최적화하지 마세요. 이를 반복하면 audit도 개발용 검증 세트가 됩니다.

좌우 TTA를 별도 실험으로 쓰려면 **학습 시작 시** `--tta hflip`을 지정합니다.
checkpoint 선택과 추론에 동일한 TTA를 적용합니다. 추론 횟수와 CPU 시간이 늘어납니다.

## 5. 같은 모델의 CSV + 최종 ZIP 만들기

```powershell
& $py make_package.py --run runs\regnet32_224 --test-dir ..\test --sample ..\sample_submission.csv --output submission_regnet32_package.zip --csv submission_regnet32.csv
```

이 명령은 학습 당시 코드 스냅샷/모델/분할/설정을 ZIP으로 묶고 새 임시 폴더에 풉니다.
**압축을 푼 코드에서 CPU 추론을 실행하여 제출 CSV를 만듭니다.**
따라서 별도의 GPU notebook 출력 CSV와 최종 ZIP이 다른 설정이 되는 일을 줄입니다.

검사 항목: 파라미터 20M 이하, ZIP 200,000,000 byte 이하, 500개의 고유 ID,
`id,prob` 헤더, sample과 ID 일치, 유효 확률, 로컬 CPU 전체 실행 600초 이하.
추론은 `pretrained=False`로 전체 `model.pt`를 로드하며 다운로드하지 않습니다.
CSV 확률은 IT관(1)의 확률입니다. 행 순서는 파일명 순이며 ID로 대응합니다.

최종 ZIP에는 학습 코드가 포함됩니다. 학습 초기 가중치까지 넣어 오프라인 재학습을
지원하려면 패키징 명령에 `--include-initial`을 추가합니다.
이 경우 `initial_model.pt`는 학습 초기화 전용이며 추론에 사용하는 모델은 여전히 하나입니다.
재학습 시 `training_config.json`의 원래 옵션에 `--init initial_model.pt`를 추가하고
새 `--out` 경로를 지정합니다. 하드웨어/라이브러리에 따른 학습 수치 차이는 가능합니다.

## 6. 제출 전 Colab CPU 확인

**로컬 CPU 통과는 Colab CPU 통과를 보장하지 않습니다.** Colab에서 런타임을 CPU로 두고
최종 ZIP을 푼 후 다음 명령을 실행하세요. 테스트 폴더도 준비되어 있어야 합니다.

```python
!python predict.py --test_dir /content/test --output /content/reproduced.csv
```

PyTorch/torchvision 버전은 `requirements_cpu.txt`에 기록됩니다. 설치가 필요하면
추론 실행 전 환경 준비 단계에서 맞추세요. 이미 같은 버전이면 재설치하지 않습니다.
`predict.py`는 버전이 다를 때 경고하고 실행합니다. 다른 버전의 수치 일치까지 보장하지는 않습니다.
CPU 장치별로 아주 작은 부동소수점 차이가 날 수 있습니다. 아래 비교로 확인하세요.

```python
import csv
import numpy as np

def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return {r['id']: float(r['prob']) for r in csv.DictReader(f)}

a = read_csv('/content/submission_regnet32.csv')
b = read_csv('/content/reproduced.csv')
assert len(a) == len(b) == 500 and a.keys() == b.keys()
ids = sorted(a)
av, bv = np.array([a[k] for k in ids]), np.array([b[k] for k in ids])
print('max absolute difference:', float(np.max(np.abs(av - bv))))
print('within 1e-6:', np.allclose(av, bv, rtol=0, atol=1e-6))
```

1e-6은 자체 점검 기준이며 대회의 공식 허용 오차가 아닙니다.
재실행 조건에 가장 가깝게 맞추려면 Colab CPU에서 생성한 CSV를 제출하고 해당 ZIP과 짝지어 보관하세요.
Colab에서 600초를 초과하면 그 ZIP으로 제출하지 마세요.

`submission_regnet32.csv`를 CSV 제출 칸에, `submission_regnet32_package.zip`을 최종 파일 칸에 제출합니다.
코드만 들어 있는 이 다운로드 키트를 최종 ZIP으로 제출하면 안 됩니다.
최종 평가 대상으로 선택한 CSV와 업로드한 ZIP의 모델/설정이 반드시 일치해야 합니다.
기존 최고 #24(0.0339)의 CSV와 ZIP도 한 쌍으로 보관하세요.
새 모델이 개선되지 않았다면 최종 선택은 기존 제출로 유지할 수 있습니다.

## 점수 해석과 다음 실험

원래 compare.py는 epoch 선택, temperature 학습, 후보 비교에 같은 582장 검증 세트를 씁니다.
첨부 로그에서 B2는 epoch 5의 select loss 0.03650이 최저이고 이후 train loss 감소에도
validation loss는 개선되지 않았습니다. 과적합 가능성은 있지만 사진별 오류를 보기 전 확정할 수 없습니다.
B0 288의 raw 검증 0.02964 → TTA/보정 후 0.02760도 Public 0.0371을 보장하지 않습니다.

Public 250장 중 정답이 1인 사진 한 장에 0.01을 예측하면 그 사진만으로
전체 평균에 약 0.01842가 더해집니다. 소수의 확신한 오답이 큰 점수 차이를 만들 수 있습니다.
아키텍처 하나를 바꾸면 자동으로 0.02 초반이 된다고 해석하지 마세요.

처음에는 이 기본 실험 하나를 완료하고 `history.csv`, `best.json`, `audit_metrics.json`을 확인합니다.
어려운 train/select 사진의 손실을 살펴 증강이나 표현력을 개선할지 판단하세요.
audit 오답을 상세히 분석해 변경에 이용했다면 이후에는 그 audit을 독립 평가로 부를 수 없습니다.
검증 분할 변경은 점수가 낮아질 때까지 반복하는 용도로 사용하지 않습니다.
추가 검증이 필요하면 사전에 정한 다른 분할에서 독립 학습을 비교하고, 결과를 모두 기록하세요.
테스트 pseudo-label, 테스트 기반 추가 학습, 수동 테스트 정답 추정/확률 수정은 하지 않습니다.

## 참고

- RegNetY-3.2GF 공식 API/가중치: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.regnet_y_3_2gf.html
- 확률 보정과 독립 자료: https://scikit-learn.org/stable/modules/calibration

이 키트의 검증 기록은 VALIDATION.txt를 참고하세요. 실제 데이터/사용자 GPU에서의
성능, 시간, Public/Private 점수는 이 환경에서 측정한 값이 아닙니다.
