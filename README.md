# E2E_Refinement_Practice

nuPlan 시뮬레이션으로 자율주행 planner 를 평가하는 실습 저장소.

ML planner 의 출력을 **Open-loop 와 Closed-loop 양쪽으로 채점**하고, 그 궤적을 **MPC 로
후처리**한 효과까지 비교한다. 실습 모델은 **PLUTO** 와 **Diffusion Planner** 두 가지이며
`model_adapter` 를 교체하여 바꾼다.

## 필수 apt 패키지 다운로드
```bash
sudo apt update
sudo apt install git curl
```

## 실습 목표

1. **ML Planner 를 검증하는 nuPlan framework 에 대해 이해하고, ML Planner 의 검증 방식에 대해
   실습한다.**
2. **Closed-loop 시뮬레이션에서 ML Planner 후처리의 필요성을 이해하고, 실제로 구현해 본다.**

목표 1 은 실습 1·2 에서, 목표 2 는 실습 3 에서 다룬다.

---

## 실습 순서

**[practice/practice0_environment_setting.ipynb](practice/practice0_environment_setting.ipynb) 부터 순서대로** 진행한다.
실행 절차와 설명은 모두 노트북에 기술되어 있으며, 본 문서는 전체 구성만을 제시한다.

| 실습 | 노트북 | 다루는 것 |
|---|---|---|
| 0 | [practice0_environment_setting.ipynb](practice/practice0_environment_setting.ipynb) | conda 환경 생성 · 패키지 · acados · MPC 솔버 빌드 · 데이터 확인 · 스모크 런 2회 |
| 1 | [practice1_nuplan_framework.ipynb](practice/practice1_nuplan_framework.ipynb) | nuPlan 프레임워크 — 시나리오가 어디서 오고, ego 가 어떻게 움직이고, 무엇을 교체할 수 있는가 |
| 2 | [practice2_ml_planner_evaluation.ipynb](practice/practice2_ml_planner_evaluation.ipynb) | ML planner(PLUTO) 장착 · Open-loop 와 Closed-loop 평가 · 영상과 지표 |
| 3 | *(준비 중)* | MPC refinement — ML 궤적 후처리와 스텝별 비교 |

실습 1 은 규칙 기반 planner 만을 사용하므로 GPU 와 acados 가 필요하지 않다.
실습 2 는 모델 추론에 GPU 를 쓰지만 acados 는 필요하지 않다. MPC 는 실습 3 에서 도입된다.

---

## 준비

실습 환경 자체(conda 환경 `e2e_refinement`, 패키지, acados)는 **실습 0 이 만든다.**
여기서는 그 노트북을 열기 위한 최소한만 갖춘다.

### 1. miniconda — 이미 `conda` 가 있으면 건너뛴다

실습 환경을 담을 conda 를 설치한다. 설치 후 셸을 다시 열어야 `conda` 명령이 잡힌다.

```bash
curl -fsSL -o /tmp/miniconda.sh \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3" && rm -f /tmp/miniconda.sh
"$HOME/miniconda3/bin/conda" init bash && exec bash
```

### 2. conda 환경 생성과 한글 폰트

실습을 진행할 가상환경을 생성하고, 시각화를 위해 필요한 한글폰트를 설치한다.

```bash
sudo apt install fonts-nanum  # 그래프 라벨이 한글이라, 없으면 빈 네모로 나온다
conda create -n e2e_refinement python=3.9.25 pip==24.0
conda activate e2e_refinement
conda install jupyterlab ipykernel   # 노트북 실행기 + 커널
```

커널 등록 후 노트북에서 **Kernel > Change Kernel > E2E Refinement** 를 선택한다.

---
### 3. 시뮬레이션을 위한 주행로그 및 모델 Weight 다운로드
다음 Google Drive 에서 data.zip 파일을 다운로드받고, 압축 해제한다.
- 디렉토리 구조는 *data/db, data/exp, data/maps, data/model*구조를 가진다.
- 기존에 존재하는 *data/* 폴더는 지우고 다운로드받은 *data/*폴더로 대체한다   
[GoogleDriveLink](https://drive.google.com/drive/folders/1hs00MrH0mUzjIZvrRQfZwG0VNuaWofE9?usp=sharing)
```
https://drive.google.com/drive/folders/1hs00MrH0mUzjIZvrRQfZwG0VNuaWofE9?usp=sharing
```
## 디렉토리 구조

```
config/                 실습에 필요한 Hydra기반 Config 모음
  planner/              refinement_planner.yaml + model_adapter/{pluto,diffusion}
  scenario_filter/      practice_scenarios / practice_single_scenario /
                        practice_random_scenarios / test14_hard
data/                   db(로그 3개) / maps / model(체크포인트) / exp(산출물)
nuplan/                 벤더링된 nuPlan devkit — 시뮬레이션 구성 요소의 설정이 여기 있다
src/                    실습 코드
  planners/refinement_planner.py    실습 2·3 의 경로계획 용 Planner
  planners/evaluator/               nuPlan 공식 지표 채점기
  planners/utils/mpc_interface.py   MPC 솔버 호출부
  feature_builders/                 모델 입력 생성 + 렌더러
Trajectory_refinement/  MPC 솔버 (acados 는 실습 0 에서 여기에 설치된다)
diffusion_planner/      Diffusion Planner 모델
practice/               실습 노트북
script/                 환경 구성 · 빌드 · 실행 스크립트
tool/                   일회용 실험 스크립트 (실습에는 사용하지 않는다)
```
