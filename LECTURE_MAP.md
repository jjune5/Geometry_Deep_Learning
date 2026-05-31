# 강의 ↔ 실습 매핑

AIMS 2022 *Geometric Deep Learning* 강의(1–12)와 이 레포의 실습 3개 대응.

## 핵심
강의의 세 가지 계산 블록에 실습이 하나씩 대응한다.

| 실습 | 주제 | 주 강의 | 토대 |
|---|---|---|---|
| **Practical 1** — GNN | 그래프·집합 | **L5–6 Graphs & Sets** | L3 |
| **Practical 2** — Group Equivariant CNN | 군(group) | **L8 Groups** | L3 |
| **Practical 3** — Geometric GNN | 기하 그래프 | **L9 Manifolds·Meshes·Geometric graphs** | L6, L3 |

## Practical 1 — `GDL100_Practical1*`
| 노트북 | 강의 |
|---|---|
| Part 1 — Cora, `GCNLayer` (노드 분류) | L5 (순열 등변, GNN 블루프린트) |
| Part 2 — ZINC, `GINLayer`/`GIN`, scatter readout | L6 (message passing, pooling) |
| Part 3 — 표현력·WL 검정·GIN | L6 (Weisfeiler–Lehman, 동형 판별) |

## Practical 2 — `Tutorial_on_Group_Convolutional_Networks*`
| 노트북 | 강의 |
|---|---|
| §1 — `C4`/`D4`, 군작용 `rotate` | L3 (군·작용·표현) |
| §2–3 — `IsotropicConv2d`/`LiftingConv2d`/`GroupConv2d`/`C4CNN` | L8 (group convolution, roto-translation, steerable) |
| 보너스 — `D4CNN` (p4m, 반사) | L8 (→ L10 게이지로 일반화) |

## Practical 3 — `[Public]_GDL100_Practical_3*`
| 노트북 | 강의 |
|---|---|
| `MPNNLayer`/`MPNNModel` | L6 (message passing) |
| `InvariantMPNNLayer`/`EquivariantMPNNLayer` (EGNN) | L3 (불변/등변) + L8 (roto-translation 등변) |
| QM9 분자 = 좌표 있는 기하 그래프 | L9 ("geometric graphs"), Seminar 1 (물리 GNN) |

## 실습이 없는 강의
L1 Introduction · L2 High-dimensional learning · L4 GDL Blueprint · L7 Grids(자료 없음) · L10 Gauges · L11 Beyond Groups · L12 Conclusions

- L3–4(불변/등변 + 블루프린트)는 모든 실습의 공통 토대.
- L10(Gauges)·L11(Beyond Groups)은 위 블록들을 일반화하는 이론 강의로 전용 실습은 없음.
