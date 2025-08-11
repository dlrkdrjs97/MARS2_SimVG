# SimVG 배치 처리 계획

## 목표
VG-RS 데이터셋의 모든 이미지에 대해 기존 `tools/demo.py`를 활용하여 visual grounding을 수행하고, 결과를 JSON 형태로 정리하기

## 입력 데이터
- **이미지**: `/home/dlrkdrjs97/workspace/code/iccv/mars2/dataset/VG-RS-images/` 내의 모든 이미지
- **쿼리**: `/home/dlrkdrjs97/workspace/code/iccv/mars2/dataset/VG-RS-question.json`
  ```json
  [
    {
      "image_path": "images\\24_0_106_0000010_250128.jpg",
      "question": "A turned-off TV"
    },
    ...
  ]
  ```

## 처리 방식
1. **기존 demo.py 활용**: 검증된 inference 파이프라인 사용
   ```bash
   CUDA_VISIBLE_DEVICES=2 python tools/demo.py \
     --img data/demo.jpg \
     --expression "A turned-off TV" \
     --config configs/demo_config.py \
     --checkpoint pretrain_weights/det_best.pth \
     --branch decoder
   ```

2. **배치 처리 스크립트 작성**:
   - VG-RS-question.json을 읽어서 각 항목에 대해
   - 해당 이미지 경로를 확인하고
   - demo.py를 subprocess로 호출하여 처리
   - 결과에서 bounding box 좌표 추출

3. **좌표 추출 방법**:
   - demo.py의 결과물에서 시각화된 이미지 대신 좌표 데이터 추출
   - 또는 demo.py를 수정하여 좌표를 직접 반환하도록 변경

## 출력 형식
```json
[
  {
    "image_path": "images\\example.jpg",
    "question": "What object is next to the red car?",
    "result": [[x1, y1], [x2, y2]]
  },
  ...
]
```

### 중요 요구사항
- `image_path`와 `question`은 원본 VG-RS-question.json과 **완전히 동일**
- `result`는 **원본 해상도**에서의 Top-left/bottom-right 좌표
- 좌표 형식: `[[x1,y1],[x2,y2]]`

## 구현 단계

### 1단계: demo.py 수정
- `--output-coords` 옵션 추가하여 좌표를 JSON으로 출력
- **기존 시각화 기능 유지**: bounding box가 그려진 이미지도 함께 생성
- 좌표와 시각화 결과 모두 생성하는 하이브리드 모드

### 2단계: 배치 처리 스크립트 작성
```python
# batch_process_demo.py 예시 구조
import json
import subprocess
import os
from tqdm import tqdm

def process_single_item(image_path, question):
    # demo.py 호출하여 좌표 추출
    cmd = [
        "python", "tools/demo.py",
        "--img", image_path,
        "--expression", question,
        "--config", "configs/demo_config.py", 
        "--checkpoint", "pretrain_weights/det_best.pth",
        "--branch", "decoder",
        "--output-coords"  # 새로 추가할 옵션
    ]
    # subprocess 실행 및 결과 파싱
    # return [[x1, y1], [x2, y2]]

def main():
    # VG-RS-question.json 로드
    # 각 항목에 대해 process_single_item 호출
    # 결과를 output.json으로 저장
```

### 3단계: 결과 검증
- 좌표가 원본 이미지 해상도 범위 내에 있는지 확인
- 몇 개 샘플에 대해 시각화하여 정확성 검증

## 장점
- **안정성**: 이미 검증된 demo.py 파이프라인 활용
- **정확성**: 좌표 스케일링 문제 해결 (demo.py에서 이미 처리됨)
- **유지보수**: 기존 코드 최소한 수정

## 예상 소요시간
- 약 22,000개 항목 × 0.2초/항목 ≈ 1.2시간 (추정)

## 실행 계획
1. `tools/demo.py`에 `--output-coords` 옵션 추가
2. `batch_process_demo.py` 작성
3. 소규모 테스트 (10개 항목)
4. 전체 데이터셋 처리
5. 결과 검증 및 시각화
