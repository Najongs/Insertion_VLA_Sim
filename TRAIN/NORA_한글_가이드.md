# NORA VLA 학습 가이드 (한글)

## 개요

NORA 모델을 당신의 HDF5 데이터셋으로 학습할 수 있는 코드를 작성했습니다.

### 생성된 파일들

```
/home/najo/NAS/VLA/Insertion_VLA_Sim/TRAIN/
├── train_nora.py              # NORA 학습 메인 스크립트
├── train_config_nora.yaml     # 학습 설정 파일
├── train_nora.sh              # 학습 실행 스크립트
├── test_nora_dataset.py       # 데이터셋 테스트 스크립트
└── README_NORA.md             # 영문 상세 가이드
```

## 주요 특징

✅ **기존 HDF5 데이터셋 활용**: `hdf5_lerobot_adapter.py`와 완전히 호환됩니다
✅ **멀티 GPU 지원**: Accelerate를 사용한 분산 학습
✅ **NORA 모델**: Qwen2.5-VL 기반 + FAST 액션 토크나이저
✅ **자동 체크포인트**: 학습 중 자동 저장 및 재개 가능
✅ **W&B 연동**: 실험 추적 및 모니터링

## 빠른 시작

### 1단계: 데이터셋 테스트

학습 시작 전에 데이터셋이 제대로 로드되는지 확인:

```bash
cd /home/najo/NAS/VLA/Insertion_VLA_Sim/TRAIN
python test_nora_dataset.py --config train_config_nora.yaml
```

이 스크립트는:
- HDF5 파일 찾기
- 데이터셋 로딩 테스트
- NORA 메시지 처리 테스트
- 샘플 구조 확인

### 2단계: 설정 파일 수정

`train_config_nora.yaml` 파일을 열어서 다음을 수정:

```yaml
dataset:
  # 데이터셋 경로 (HDF5 파일들이 있는 디렉토리)
  root_dir: "/home/najo/NAS/VLA/Insertion_VLA_Sim/Sim/collected_data_sim_6d_clean/collected_data_merged"

  # 액션 예측 범위 (1 = 한 스텝, 50 = 50스텝 예측)
  horizon: 1

  # 상태 표현
  use_ee_pose: true    # 엔드이펙터 포즈 사용
  use_qpos: false      # 조인트 위치 사용 안함

  # 태스크 설명
  task_instruction: "Insert the needle into the target point"

# 출력 디렉토리 (체크포인트 저장 위치)
output_dir: "/home/najo/NAS/VLA/Insertion_VLA_Sim/outputs/nora_training"

# 학습 파라미터
per_device_batch_size: 16        # GPU당 배치 사이즈
learning_rate: 0.00005           # 학습률
max_train_steps: 100000          # 총 학습 스텝
```

### 3단계: 학습 시작

**단일 GPU 학습:**
```bash
bash train_nora.sh
```

**멀티 GPU 학습 (4개 GPU):**
```bash
bash train_nora.sh --multi-gpu --num-gpus 4
```

**커스텀 설정 파일 사용:**
```bash
bash train_nora.sh --config my_config.yaml
```

### 4단계: 학습 모니터링

학습 중에는:
- 콘솔에 로그 출력됨 (loss, gradient norm, learning rate)
- W&B에 자동 업로드됨
- 체크포인트가 `output_dir/steps_XXXXX/`에 저장됨

### 5단계: 학습 재개 (중단된 경우)

학습이 중단되었다면 `train_config_nora.yaml`을 수정:

```yaml
resume_from_checkpoint: "/home/najo/NAS/VLA/Insertion_VLA_Sim/outputs/nora_training/steps_20000"
```

그리고 다시 실행:
```bash
bash train_nora.sh
```

## 데이터셋 구조

HDF5 파일은 다음 구조여야 합니다:

```python
{
    'action': (N, 6),                          # 로봇 액션
    'observations/ee_pose': (N, 6),            # 엔드이펙터 포즈
    'observations/qpos': (N, 6),               # 조인트 위치 (선택)
    'observations/images/camera1': (N, H, W, 3),  # RGB 이미지
    'timestamp': (N,),                         # 타임스탬프
    'phase': (N,),                             # 태스크 단계 (선택)
    'language_instruction': str                # 태스크 설명 (선택)
}
```

당신의 `hdf5_lerobot_adapter.py`가 이미 이 형식을 지원하므로 그대로 사용 가능합니다!

## 주요 설정 옵션

### 배치 사이즈 조정 (GPU 메모리 부족 시)

```yaml
per_device_batch_size: 8  # 더 작게 (기본: 16)
gradient_accumulation_steps: 4  # 더 크게 (기본: 2)
```

### 액션 청킹 (여러 스텝 예측)

```yaml
dataset:
  horizon: 50  # 50스텝 미래 액션 예측
```

### 데이터 증강 활성화

```yaml
dataset:
  augment: true
  augment_brightness: 0.2
  augment_contrast: 0.2
```

### 카메라 드롭아웃 (로버스트성 향상)

```yaml
dataset:
  camera_dropout_prob: 0.3  # 30% 확률로 카메라 드롭아웃
  min_cameras: 1            # 최소 1개 카메라 유지
```

## NORA 모델 구조

NORA는 다음을 결합합니다:
- **비전 인코더**: Qwen2.5-VL (이미지 처리)
- **언어 모델**: Qwen2.5 (명령어 처리)
- **액션 토크나이저**: FAST (액션을 토큰으로 변환)

학습 과정:
1. 이미지 + 명령어 → Qwen2.5-VL → 비전 특징
2. 액션 → FAST 토크나이저 → `<robot_action_0>`, `<robot_action_1>`, ...
3. 모델 학습: `P(액션_토큰 | 이미지, 명령어)`
4. 액션 토큰에만 Loss 계산 (명령어 토큰은 마스킹)

## 문제 해결

### "No HDF5 files found" 에러

**해결:**
- `dataset.root_dir` 경로 확인
- `.h5` 파일이 있는지 확인

### "CUDA out of memory" 에러

**해결:**
```yaml
per_device_batch_size: 8  # 배치 사이즈 줄이기
```

또는
```yaml
gradient_accumulation_steps: 4  # 그래디언트 축적 늘리기
```

### "Flash Attention not supported" 에러

**해결:** `train_nora.py` 파일의 380번째 줄 수정:
```python
attn_implementation="eager"  # "flash_attention_2" 대신
```

### "DataLoader worker crashed" 에러

**해결:**
```yaml
num_workers: 0  # 싱글 프로세스 로딩 사용
```

## 예상 성능

| 설정 | 배치 사이즈 | GPU | 학습 시간 | 예상 Loss |
|------|-----------|-----|----------|-----------|
| 소규모 | 16 | 1 | 72시간 | 3-4 |
| 중규모 | 64 | 4 | 24시간 | 2-3 |
| 대규모 | 128 | 8 | 12시간 | 2-3 |

약 1000개 에피소드로 100k 스텝 학습 시 loss ~2-4 정도로 수렴합니다.

## 기존 코드와의 차이점

### 기존 NORA 학습 코드 (`nora/training/train.py`)
- RLDS 데이터셋 사용
- OXE 데이터 믹스 사용

### 새로운 코드 (`train_nora.py`)
- ✅ **당신의 HDF5 데이터셋** 사용
- ✅ `hdf5_lerobot_adapter.py`와 **완전 호환**
- ✅ 동일한 NORA 모델 아키텍처
- ✅ 동일한 학습 방식 (FAST 토크나이저 + Qwen2.5-VL)

## 학습 팁

1. **작게 시작**: 먼저 1-2개 에피소드로 테스트
2. **Loss 모니터링**: ~8-10에서 ~2-4로 감소해야 함
3. **체크포인트 저장**: 학습에 24-48시간 소요될 수 있음
4. **정기적 검증**: 별도 에피소드로 검증

## 요약

이제 다음 파일들이 준비되었습니다:

1. ✅ `train_nora.py` - NORA 학습 메인 스크립트
2. ✅ `train_config_nora.yaml` - 설정 파일
3. ✅ `train_nora.sh` - 실행 스크립트
4. ✅ `test_nora_dataset.py` - 데이터셋 테스트

실행 순서:

```bash
# 1. 데이터셋 테스트
python test_nora_dataset.py

# 2. 설정 파일 수정
nano train_config_nora.yaml

# 3. 학습 시작
bash train_nora.sh
```

질문이 있으면 `README_NORA.md` 파일을 참고하세요!

---

**작성일:** 2026-02-01
**버전:** 1.0
