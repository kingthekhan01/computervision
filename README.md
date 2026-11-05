# 영상처리 과제 (11/06)

## 1. Morphology — `morphology_1106.py`

OpenCV 내장 함수(`cv.erode`, `cv.dilate`) 없이 반복문으로 직접 구현한 형태학 연산입니다.

| 연산 | 입력 영상 | 커널 |
|---|---|---|
| Opening (침식 → 팽창) | `12345.png` (A + 점 노이즈) | 3×3 사각형 |
| Closing (팽창 → 침식) | `closer.png` (A 안쪽 구멍) | 3×3 사각형 |
| Dilation | `12345.png` | 3×3 십자형 |
| Erosion | `12345.png` | 3×3 십자형 |

- 영상 바깥은 0으로 패딩
- 실행하면 입력/결과 영상이 창으로 표시되고, 아무 키나 누르면 종료

### 실행 결과

| 입력 | Opening |
|---|---|
| ![](12345.png) | ![](results/a_opening.png) |

| 입력 | Closing |
|---|---|
| ![](closer.png) | ![](results/b_closing.png) |

| Dilation | Erosion |
|---|---|
| ![](results/a_dilation.png) | ![](results/a_erosion.png) |

## 2. KNN 분류 — `KNN1106.py`

CIFAR-10 이미지를 픽셀 벡터(3072차원)로 펼쳐서 직접 구현한 KNN으로 분류합니다.

- 데이터: 학습 5,000장 / 테스트 1,000장 (클래스 비율 유지 샘플링, `random_state=42`)
- 거리: 유클리드(L2), 맨해튼(L1)
- k: 1, 3, 5
- 5-fold 교차검증으로 평균 정확도가 가장 높은 (거리, k) 조합을 고른 뒤 테스트셋으로 최종 평가
- 지표: Accuracy, Precision / Recall / F1 (macro 평균)

CIFAR-10 데이터는 처음 실행할 때 `data/` 폴더에 자동으로 다운로드됩니다.
순수 Python 반복문으로 거리를 계산해서 실행에 시간이 꽤 걸립니다.

### 실행 결과

**5-fold 평균 검증 결과**

| 거리 | k | Accuracy (mean ± std) | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Euclidean | 1 | 0.2746 ± 0.0114 | 0.3221 | 0.2746 | 0.2641 |
| Euclidean | 3 | 0.2844 ± 0.0057 | 0.3343 | 0.2844 | 0.2698 |
| Euclidean | 5 | 0.2882 ± 0.0104 | 0.3569 | 0.2882 | 0.2687 |
| Manhattan | 1 | 0.2924 ± 0.0095 | 0.3350 | 0.2924 | 0.2873 |
| Manhattan | 3 | 0.3018 ± 0.0095 | 0.3439 | 0.3018 | 0.2928 |
| **Manhattan** | **5** | **0.3172 ± 0.0130** | **0.3671** | **0.3172** | **0.3037** |

→ 평균 검증 정확도가 가장 높은 **Manhattan, k=5** 선택

**최종 테스트 결과 (Manhattan, k=5)**

| Accuracy | Precision | Recall | F1-score |
|---|---|---|---|
| 0.2980 | 0.3389 | 0.2980 | 0.2848 |

전체 콘솔 출력(fold별 결과 포함): [`results/knn_result.txt`](results/knn_result.txt)

## 실행 방법

```bash
pip install -r requirements.txt
python morphology_1106.py
python KNN1106.py
```
