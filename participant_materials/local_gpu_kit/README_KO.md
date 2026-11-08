# AI Mini Challenge — RTX 3070 Ti 로컬 GPU 실험 키트

현재 Colab B0(224) + Horizontal-Flip TTA 결과: Validation Log Loss **0.03186**, Public **0.0339**. 아래 명령은 **기존 최고 모델을 덮어쓰지 않으며**, 제공된 2,906장 이미지의 고정 검증 분할(Seed 42, 80/20)을 그대로 사용합니다.

## 0. 준비물과 폴더

이 ZIP을 압축 해제해서, 기존 `participant_materials/participant_materials/` 폴더 안에 `local_gpu_kit/` 디렉터리가 생기도록 놓습니다.

```text
participant_materials/participant_materials/
  train/6e/*.jpg
  train/it/*.jpg
  test/*.jpg
  sample_submission.csv
  baseline_model.pt    <-- Google Drive/AI_Challenge/model.pt 내려받아 이름 변경
  AI_Mini_Challenge_EfficientNet_B0_(1).ipynb   <-- 원래 학습 노트북 (선택; 최종 ZIP에 포함 권장)
  local_gpu_kit/
    train.py
    compare.py
    model.py
    predict.py
    make_package.py
    README_KO.md
```

## 1. Windows PowerShell / Python 3.12 + GPU 가상환경

`local_gpu_kit`에서 실행합니다. `py -3.12`가 없다면 `py -3.11` 또는 설치된 `python`을 사용하세요.

```powershell
nvidia-smi
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install numpy pandas pillow scipy scikit-learn matplotlib
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

NVIDIA 드라이버가 CUDA 12.8 휠을 지원하지 않는다면 드라이버를 갱신하거나, PyTorch 공식 설치 안내에서 지원 가능한 CUDA 빌드를 선택하세요. PyTorch 휠에는 CUDA 런타임이 포함되므로 일반적인 학습만 한다면 별도 CUDA Toolkit 설치는 필수는 아닙니다.

## 2. 기존 B0 가중치에서 288×288 미세조정

```powershell
.\.venv\Scripts\python.exe train.py --data .. --arch b0 --init ..\baseline_model.pt --size 288 --epochs 8 --lr 2e-5 --batch-size 16 --out runs\b0_288
```

학습 중 `CUDA out of memory`라면 `--batch-size 8`로 다시 시작하세요. 기존 `baseline_model.pt`는 수정하지 않습니다. 가장 낮은 **검증 원본 CrossEntropy** 모델을 `runs\b0_288\best_model.pt`에 저장합니다. 6~8 epoch에서도 val이 악화되면 더 돌리지 않고 기존 후보를 유지하세요.

## 3. B2 신규 학습 (ImageNet 사전학습 가중치 사용)

```powershell
.\.venv\Scripts\python.exe train.py --data .. --arch b2 --pretrained --size 288 --epochs 12 --lr 1e-4 --batch-size 12 --out runs\b2_288
```

학습 중 사전학습 가중치 다운로드는 허용됩니다. **최종 추론은 인터넷 다운로드를 하지 않습니다.** GPU 메모리가 부족하면 배치 8로 낮추세요.

## 4. 성능 비교 및 후보 JSON 자동 생성

B0 288만 학습했을 때는 `--b2 ...`를 생략해도 됩니다.

```powershell
.\.venv\Scripts\python.exe compare.py --data .. --baseline ..\baseline_model.pt --b0-288 runs\b0_288\best_model.pt --b2 runs\b2_288\best_model.pt --output candidates
```

출력: `candidates\validation_comparison.csv`와 각 실험 조합의 JSON 파일. **기존 B0(224)의 TTA Log Loss 0.03186**보다 낮은지 비교하세요. 단, 검증셋이 학습 중 체크포인트 선택과 온도 보정에도 사용되므로 결과가 조금 낙관적일 수 있습니다. 많은 조합을 최적화하지 마세요.

앙상블 후보는 파라미터 합계가 20M 이하인 조합만 생성됩니다. TTA는 모델의 파라미터 수를 증가시키지 않지만 CPU 시간을 증가시킵니다. 검증 결과 1위가 CPU 10분 제한을 초과하면 2위나 `..._orig` 후보를 사용하세요.

## 5. 선택한 후보로 최종 CSV + ZIP 생성·재현성 점검

예시: B0(224) + B2(288)의 TTA 평균이 선택됐다면

```powershell
.\.venv\Scripts\python.exe make_package.py --candidate candidates\ensemble_baseline_b2_288.json --test-dir ..\test --output ..\submission_b0b2_package.zip --csv ..\submission_b0b2.csv --notebook "..\AI_Mini_Challenge_EfficientNet_B0_(1).ipynb"
```

학습 노트북의 파일명이 다르면 `--notebook` 경로를 수정하세요. (누락되면 매개변수 자체를 생략할 수 있지만 기존 초기 모델을 재현하려면 포함 권장.)

이 스크립트는 **완성 ZIP을 별도 임시 폴더에 풀고, 인터넷 다운로드 없이 CPU로 500장을 실제 추론**하여 CSV를 생성합니다. ZIP에는 `model.py`, `predict.py`, `train.py`, `submission_config.json`, 모든 가중치, `README.txt`, 선택적 원본 학습 노트북이 들어갑니다.

단, **로컬 PC의 CPU 시간이 600초 이내여도 Colab CPU는 더 느릴 수 있습니다.** 최종 제출 전에 Colab CPU에서 ZIP을 풀어 동일한 명령으로 500장 추론 시간과 CSV 일치를 확인하세요.

## 6. 중요한 대회 규칙

- `train/` 2,906장만 학습. `test/` 500장은 추론 전용. pseudo-labeling 금지.
- 제출 CSV는 정확히 헤더 `id,prob`, 테스트 500개 ID, IT관(label 1) 확률.
- 20M 총 파라미터, 200MB ZIP, Colab CPU 600초 이내, 외부 다운로드 금지.
- **현재 점수 0.0339의 `submission_tta.csv`와 동일한 ZIP은 별도로 백업**하고 절대 덮어쓰지 마세요.
- Public 점수 0.02 초반은 보장할 수 없습니다. 반복적인 Public 기반 튜닝보다 검증 성능과 독립 재현성 우선.
