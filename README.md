# Geometry_Deep_Learning

**Geometric Deep Learning** (AMMI / GDL100) 강의 실습 노트북 풀이입니다.

각 실습마다 노트북이 **두 개**씩 있습니다.

| 실습 | 빈 노트북 (원본) | 정답본 |
|---|---|---|
| **1 — GNN** (Cora GCN, ZINC GIN, 표현력 / 1-WL) | `GDL100_Practical1.ipynb` | `GDL100_Practical1_solution.ipynb` |
| **2 — 그룹 등변 CNN** (C₄/D₄ 군론, lifting & group convolution, 회전 MNIST) | `Tutorial_on_Group_Convolutional_Networks_AMMI_Geometric_Deep_Learning_Course.ipynb` | `..._solution.ipynb` |
| **3 — 기하학적 GNN** (QM9, 불변/등변 메시지 패싱 / EGNN) | `[Public]_GDL100_Practical_3.ipynb` | `[Public]_GDL100_Practical_3_solution.ipynb` |

각 실습은 아래 **네 가지** 파일로 제공됩니다(`X` = 노트북 이름):

| 파일 | 설명 |
|---|---|
| `X.ipynb` | 빈 원본 (배포본 그대로) |
| `X_ko.ipynb` | **빈 원본의 한국어판** — 설명·주석은 한글, 코드 빈칸은 그대로(직접 채우는 용) |
| `X_solution.ipynb` | 정답본 (영문) |
| `X_solution_ko.ipynb` | **정답본의 한국어판** — 설명·증명·주석까지 전부 한글 |

한국어판은 모든 마크다운(과제 설명·증명)과 코드 주석/독스트링을 한글로 옮기되,
**LaTeX 수식과 코드 로직은 그대로 유지**했습니다. 코드는 토큰 단위로 원본과 대조하여
한 글자도 바뀌지 않았음을 확인했고, 정답본 한국어판은 직접 실행해 동일하게 동작함을 검증했습니다.

## 구현 내용

**실습 1 — GNN**
- `GCNLayer` (대칭 정규화 `D̃^{-1/2} Ã D̃^{-1/2} X W`), `create_mini_batch` (블록 대각 배치),
  `scatter` 그래프 단위 readout, `GINLayer` (`MLP(AX + (1+ε)X)`), `SimpleGIN` (sum readout),
  JK 방식 전체 `GIN` (층별 readout concat).
- Part 3 (표현력): GIN이 구분하지 못하는 데이터셋(삼각형 2개 vs 육각형), GIN + **Random Node
  Initialisation**, 노드별 삼각형 개수를 쓰는 **메시지 패싱 너머** 모델, 그리고 특징 증강이
  vanilla GIN보다 엄밀히 더 강력함을 보이는 증명.

**실습 2 — 그룹 등변 CNN**
- 순환군 `C4`, 이면체군 `D4`, `p4` 작용 `rotate_p4`, `IsotropicConv2d`, `LiftingConv2d`,
  `GroupConv2d`, `C4CNN`, `IsotropicCNN`, **등변 BatchNorm**, `C4CNNWithBatchNorm`.
- **보너스:** 회전과 반사 모두에 불변인 `D4`/`p4m` 풀이 (`D4LiftingConv2d`, `D4GroupConv2d`, `D4CNN`).

**실습 3 — 기하학적 GNN**
- `CoordMPNNModel`, 회전/이동 단위 테스트, 거리 기반 `InvariantMPNNLayer`/`InvariantMPNNModel`,
  EGNN 방식 **등변** `EquivariantMPNNLayer`/`FinalMPNNModel` (특징은 불변, 좌표는 등변).

## 검증

정답본의 모든 알고리즘 코드는 CPU에서 합성 입력으로 **직접 실행하여 검증**했습니다
(`torch`, `torch_geometric`, `torch_scatter` — GPU·데이터셋 불필요):
- 실습 1: `unit_test_mini_batch` (배치 vs 개별 그래프 일치) 통과, Part 3 표현력 실험에서
  예상된 차이 재현 (vanilla GIN ≈ 무작위, GIN+RNI ≈ 0.97).
- 실습 2: lifting/group-conv 등변성과 전체 네트워크 회전(보너스는 반사까지) 불변성을 포함한
  모든 내장 `assert` 통과.
- 실습 3: 순열·회전·이동 불변/등변 단위 테스트가 모두 기대대로 동작.

학습 셀(Cora / ZINC / QM9 / 회전 MNIST)은 GPU가 있는 Google Colab에서 실행하면 됩니다.

각 실습이 강의 1–12 중 어디에 대응하는지는 [`LECTURE_MAP.md`](LECTURE_MAP.md) 참고.
