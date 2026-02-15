# `run_nav_ovon_omni.py` 상세 분석

이 문서는 `infer_ovon_slowfast/run_nav_ovon_omni.py`를 **실행 순서 기준으로 사실상 한 줄씩** 설명하고, 연결 모듈/데이터 포맷까지 정리한 문서입니다.

---

## 1) 스크립트의 역할 요약

`run_nav_ovon_omni.py`는 OVON(Object-goal Vision-and-Language Navigation) 평가를 위해:

1. 데이터셋 split(`val_seen`, `val_seen_synonyms`, `val_unseen`)을 순회하고,
2. 각 episode에서 Habitat 시뮬레이터를 초기화한 뒤,
3. 360도 회전 관측 + frontier(미탐색 경계) 후보를 수집하고,
4. Qwen 기반 Omni 모델로 다음 목표점(혹은 최종 정지점)을 결정하며,
5. `A-star` 또는 `point-goal` 방식으로 이동,
6. episode별 `SR`, `SPL`을 계산해 JSON 결과와 비디오를 출력합니다.

---

## 2) 줄 단위 해설 (`run_nav_ovon_omni.py`)

> 코드 라인이 길고 반복 구간이 있어, 동일 패턴 반복문 내부는 “해당 줄 + 동작” 형태로 최대한 세밀하게 설명했습니다.

### 2.1 import 및 상수 정의

- **1행**: split별/카테고리별 통계 집계를 위해 `defaultdict` 임포트.
- **2행**: 파일/경로 작업용 `os`.
- **3행**: Habitat top-down map 유틸.
- **4행**: `Simulator as Sim`을 임포트하지만 실제 본문에서는 직접 사용되지 않음.
- **5행**: 결과/데이터 파일 로딩용 `json`.
- **6행**: Habitat Simulator 기능 전반 사용.
- **7행**: 수치 계산용 `numpy`.
- **8행**: `TopDownMap` 임포트하지만 본문 직접 사용은 없음.
- **10행**: frontier 탐색 관련 핵심 함수들 임포트.
- **11행**: `get_simulator` 임포트(현재 파일 내 직접 사용 없음).
- **12행**: OpenCV 기반 프레임/맵 시각화.
- **13행**: 샘플 섞기용 `random`.
- **14행**: 스크립트 내부 공통 유틸(인자 파싱, 로그, 데이터 로딩, 시뮬레이터 생성, 비디오 제작).
- **15행**: Qwen 추론 유틸(`getresult`, `Bank`, `get_result_fast`).
- **16행**: Torch.
- **17행**: `numpy` 중복 임포트(기능상 문제는 없지만 중복).
- **18행**: Qwen2.5-VL 모델/프로세서 로더.

- **20~28행**: 데이터/출력/설정 파일 경로 하드코딩.
  - episode 샘플 json: `ovon_full_set.json`
  - split별 navigation gzip json 루트
  - HM3D scene(.basis.glb) 루트
  - 결과 json, 로그, 비디오 저장 경로
  - 시뮬레이터/에이전트 YAML 경로
- **30행**: `decision_num_min` 선언되나 실제 사용되지 않음.
- **31행**: 가시 반경(미터) 기반 탐색 파라미터.
- **32행**: `generated_avi_files` 선언되나 본 파일에서는 사용되지 않음.
- **33~34행**: Qwen processor의 pixel 제한.
- **35행**: top-down map 해상도(512).
- **36행**: 재현성 seed.

### 2.2 인자 파싱/모델 로딩/데이터 로딩

- **38행**: `add_arguments()`로 CLI 파싱.
- **39~42행**: `pattern`, `model_path`, `name`, `fast_type` 추출.
  - 주의: `utils.py`의 parser에는 `--fast_type` 정의가 없어 현재 구조상 불일치 가능성이 큼.
- **44행**: 시각화 활성화.
- **45~48행**: 수동 디버깅용 예시 주석.
- **50행**: split 리스트 + navigation 데이터 로딩.
- **51~53행**: Qwen 모델 로드(`bfloat16`, `device_map="auto"`, `flash_attention_2`).
- **54행**: processor 로드.

### 2.3 기존 결과 이어쓰기 및 episode 필터링

- **57~60행**: 결과 파일 있으면 이어쓰기, 없으면 split별 빈 리스트 초기화.
- **62행**: 평가 대상 episode 목록(`ovon_full_set.json`) 로딩.
- **63~65행**:
  - split별로 이미 처리된 `(scan_id, episode_index)` 집합 생성,
  - data_set에서 완료분 제거(재실행 시 중복 평가 방지).

### 2.4 split/episode 루프 진입

- **68행**: split 루프 시작.
- **69행**: split 시작 시 랜덤시드 재설정.
- **70~83행**: `pattern == 'all'`이면 navigation 데이터 전체를 순회해 episode 목록 수동 구성.
  - **71행**: 여기서 `split_list=[]`로 재할당하는 부분은 변수명 충돌 위험이 큼(원래 split 이름 목록과 같은 이름 사용).
- **84행**: `all`이 아니면 `data_set[split]` 사용.
- **86행**: episode 순서 랜덤 셔플.
- **87행**: episode 루프 시작.

### 2.5 episode 메타/환경 초기화

- **89행**: `get_info`로 scene/episode 정보 추출.
- **90행**: 로그 파일 경로 생성.
- **92행**: 현재 object_category의 goal(view point 포함) 불러옴.
- **94행**: `get_sim_agent`로 Habitat sim/agent/pathfinder 및 시작 pose 세팅.
- **96행**: 시뮬레이터로부터 top-down occupancy map 생성.
- **97행**: fog-of-war 마스크(탐색 상태) 0으로 초기화.
- **98행**: frontier 영역 최소 임계(9m)를 픽셀로 변환.
- **99행**: 가시거리 픽셀 변환.

- **102행**: 의사결정 횟수 카운터.
- **103행**: 총 행동 스텝 카운터.
- **104행**: 전역 기록(이미지/프론티어맵/state).
- **105행**: 이번 goto 구간 임시 기록.
- **107행**: 이전 state 초기화(이동거리 누적용).
- **108행**: 누적 거리.
- **109행**: 방문 frontier 셋.
- **113행**: `Bank` 초기화(`goto_sample_interval=1`).
- **114~115행**: 과거 color/agent 변수 선언(현 코드에서 사실상 미사용).

### 2.6 메인 탐색 루프(`while total_steps < 4000`)

- **116행**: 안전 상한 4000 스텝.
- **117~119행**: 직전 goto 프레임이 있으면 `bank`에 축적 후 임시 리스트 초기화.

#### (A) 360도 회전 관측 수집

- **121행**: 회전 단계 color/state 리스트 초기화.
- **122행**: 좌회전 12번(30도×12=360도).
- **124~136행**: 매 회전 step마다
  - `sim.step('turn_left')`
  - RGB 추출/저장,
  - agent state 저장,
  - 현재 pose 기반으로 fog-of-war 업데이트,
  - frontier 후보 검출,
  - 시각화 맵 저장.
- **137행**: 이번 회전 데이터 `bank.add(..., 'spin')`.
- **138~141행**: 회전 프레임/상태를 전역 리스트에 병합.

#### (B) frontier 좌표 변환 + 모델 의사결정

- **142~146행**: frontier가 비어있지 않으면
  - `(y,x)` 픽셀 축 뒤집기,
  - 픽셀 → 지도 좌표(월드 좌표) 변환.
- **148행**: 이미 방문한 frontier(반올림 좌표 키) 제거.
- **149~157행**: `getresult(...)` 호출
  - 입력: spin 이미지 bank, frontier 목록, 현재 state, object category 등
  - 출력: `target_position`, `is_final_decision`, `output_texts`.
- **159행**: 콘솔 출력.
- **160행**: bank 초기화.
- **161행**: 로그 기록.
- **162행**: decision 카운터 증가.

#### (C-1) `A-star` 분기

- **163행**: `fast_type == 'A-star'`일 때.
- **164~173행**:
  - 최종 결정이 아니면 방문 frontier 셋에 목표점 추가,
  - 현재 navmesh island에서 목표점 snap,
  - GreedyGeodesicFollower로 action list 계산.
- **174~203행**: 경로 실패 예외 처리
  - 후보 frontier를 셔플해 대체 목표 재시도,
  - 성공 시 대체 경로 채택,
  - 전부 실패하면 episode 루프 탈출.

- **205행**: goto 임시 버퍼 초기화.
- **206~207행**: action list가 `None`이면 탈출.
- **208~226행**: 경로 action 실행 루프
  - 시뮬레이터 step,
  - RGB 프레임 저장,
  - fog-of-war/프론티어 갱신,
  - 총 스텝/누적이동거리 갱신,
  - frontier 시각화 프레임 저장.
- **227행**: state 기록 합치기.
- **229~230행**: 실제 이동 없었으면 탈출.
- **231~232행**: 마지막 관측/state를 보관.
- **233~234행**: 모델이 최종결정이면 episode 종료.

#### (C-2) `point-goal` 분기

- **235행**: `fast_type == 'point-goal'`.
- **236행**: goto 버퍼 초기화.
- **237~238행**: action list `None`이면 탈출.
- **239행**: 내부 반복 카운터.
- **241행**: 목표점까지 거리>0.1m && 반복<200 동안 반복.
- **244~256행**: `get_result_fast(...)` 호출
  - 최근 goto 히스토리 + 현재 tri-view + target으로
  - 다음 waypoint를 행동형으로 예측.
- **257행**: count 로그.
- **259~262행**: `'000'` 반환 시 정지 처리.
- **263~266행**: 현재 state/위치 로그.
- **268~294행**: 관측 저장/맵 갱신/거리 및 step 누적/시각화 마킹/`bank` 축적.
- **295~297행**: 목표점 근접(0.25m)이면서 최종결정 아님 → 방문 frontier 등록 + `rot=0`.
- **298~299행**: 최종결정 시 종료.

### 2.7 에피소드 종료 후 지표 계산

- **300행**: 종료 시점 agent state 확보.
- **301~305행**: 목표 object의 모든 `view_points` 좌표 추출.
- **307~316행**: 시작점→목표군 최소 geodesic 거리 계산(없으면 `inf`).
- **318~325행**: 현재점→목표군 최소 geodesic 거리 계산(없으면 `inf`).
- **327~336행**: SR/SPL 계산
  - 시작거리 `inf`: SR/SPL=1(특수 처리)
  - 현재거리 `inf`: SR/SPL=0
  - 일반: `SR = agent_end_geo_distance <= 1`, `SPL = SR * d* / max(d*, path_len)`.
- **338~339행**: 시각화 on이면 영상 생성.
- **341행**: 결과 append.
- **342행**: episode 요약 로그.
- **343~344행**: 결과 json 즉시 flush 저장.

### 2.8 split 평균 통계 출력

- **346행**: split별 후처리 루프.
- **347~349행**: 누적 변수/카테고리 통계 딕셔너리 초기화.
- **350~356행**: 결과를 합산.
- **358~360행**: split 평균 SR/SPL 출력.
- **362~365행**: 카테고리별 평균 SR/SPL 출력.

---

## 3) 이 스크립트와 연결된 내부 파일(의존 관계)

## 3.1 직접 import되어 실제 호출되는 모듈

1. **`infer_ovon_slowfast/utils.py`**
   - `add_arguments`: CLI 파싱
   - `load_navigation_data`: gzip episode 데이터 로딩
   - `get_info`: scene 경로/episode 추출
   - `make_log_file`, `log_message`: 로깅
   - `get_sim_agent`: HabitatSimulator 생성 + 시작 pose 세팅
   - `make_video`: episode 시각화 영상 생성

2. **`infer_ovon_slowfast/qwen_utils.py`**
   - `Bank`: spin/goto 이미지 메모리 뱅크
   - `getresult`: frontier 선택/최종 결정 여부 생성형 추론
   - `get_result_fast`: point-goal 모드의 빠른 waypoint 예측

3. **`infer_ovon_slowfast/frontier_utils.py`**
   - `reveal_fog_of_war`: 시야 내 탐색 가능 영역 갱신
   - `detect_frontier_waypoints`: frontier 후보 추출
   - `map_coors_to_pixel`, `pixel_to_map_coors`: 좌표계 변환
   - `get_polar_angle`, `convert_meters_to_pixel`: 각도/스케일 유틸

4. **`infer_ovon_slowfast/simulator.py`** (간접)
   - `utils.get_sim_agent` 내부에서 `HabitatSimulator` 사용.
   - 실제 센서 구성(`color_sensor`, `color_sensor_left/right`, `depth_sensor`)은 이 파일이 담당.

## 3.2 설정 파일

- **`infer_ovon_slowfast/config/habitat/goat_sim_config.yaml`**
  - navmesh/scene/agent 물리 파라미터.
- **`infer_ovon_slowfast/config/habitat/goat_agent_config.yaml`**
  - RGB 전방/좌/우 센서, depth 센서 해상도/FOV/포즈.

## 3.3 실행 스크립트

- **`infer_ovon_slowfast/eval_ovon_slowfast.sh`**
  - `PATTERN`, `MODEL_PATH`, `NAME`, `TYPE`를 지정해 본 스크립트를 실행.
  - 다만 현재 Python parser 정의와 `--fast_type` 이름 불일치 가능성 존재.

---

## 4) 입력 데이터 포맷(코드 기반 역추론)

실제 대용량 데이터는 레포에 포함되지 않았지만, 코드에서 요구하는 스키마는 다음과 같습니다.

### 4.1 `ovon_full_set.json` (`data_set_path`)

최상위 키는 split:

```json
{
  "val_seen": [
    {
      "scan_id": "xxxx-xxxx",
      "episode_index": 0,
      "object_category": "chair"
    }
  ],
  "val_seen_synonyms": [...],
  "val_unseen": [...]
}
```

- `run_nav_ovon_omni.py`는 이 파일을 읽어 처리 대상 episode를 고르고,
- 이미 `output_path`에 있는 `(scan_id, episode_index)`는 제거합니다.

### 4.2 split별 navigation gzip json (`navigation_data_path/<split>/content/*.json.gz`)

`utils.load_navigation_data` 기준 필수 구조:

- 루트에 `episodes`
- 루트에 `goals_by_category`
- `episodes[i]` 내부에 최소:
  - `start_position`: `[x, y, z]`
  - `start_rotation`: quaternion 계열(코드에 바로 set 가능 타입)
  - `object_category`
  - `episode_id`
- `goals_by_category[category]` 내부 각 goal에
  - `view_points`
  - `view_points[*]["agent_state"]["position"]`

### 4.3 HM3D scene 파일

`get_info`가 scene 경로를 다음처럼 만듭니다:

```text
{hm3d_data_base_path}/{scene_id}/{clean_scene_id}.basis.glb
```

즉 `.basis.glb`가 있어야 시뮬레이터가 정상 생성됩니다.

---

## 5) 출력물 포맷

1. **결과 JSON (`output_path`)**
   - split별 배열에 아래 객체 append:
   ```json
   {
     "scan_id": "...",
     "episode_index": 12,
     "sr": 0,
     "spl": 0.0,
     "object_category": "sofa"
   }
   ```

2. **로그 파일 (`log_dir/<name>/...log`)**
   - 모델 출력, fallback 경로, 종료 통계 등 텍스트 로그.

3. **비디오 (`video_path`)**
   - 원본 avi 생성 후 mp4 변환 시도(`ffmpeg`).
   - 좌측 RGB + 우측 frontier/topdown 오버레이 결합 영상.

---

## 6) 실행 플로우를 한 문장으로 정리

`episode 로딩 → 360도 회전으로 관측/프론티어 구축 → Qwen이 목표점 또는 final found를 결정 → A-star/point-goal 이동 반복 → 종료 시 SR/SPL 계산 및 결과 저장`.

---

## 7) 코드에서 눈에 띄는 주의점(디버깅 시 중요)

1. `utils.add_arguments`와 실제 사용 인자명이 어긋나 있음(`fast_type` vs `type`).
2. `for split in split_list` 내부에서 `split_list=[]` 재사용(변수명 충돌).
3. 중복 import/미사용 변수(`Sim`, `TopDownMap`, `decision_num_min`, `generated_avi_files` 등).
4. `enable_visualization=True`면 디스크 I/O가 매우 많아져 속도 영향 큼.

