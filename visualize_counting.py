"""
visualize_counting.py - Traffic Counting Visualization

Skia rendering, BoTSORT tracking, intersection-based counting.
여러 개의 YOLO 모델을 동시에 돌려 좌/우로 이어붙인 비교 영상을 생성할 수 있음
(ACTIVE_DETECTORS 길이 >= 2 이면 자동 concat 모드).

Single Source of Truth: configs/{video_stem}.json
  각 영상의 카운팅 라인(polylines_lst)과 ROI(roi_polygon)는 단일 config 파일에만 저장됨.
  drawing_phase 에서 생성/편집, processing_phase 에서 읽기.

Directory Layout:
  input/               원본 영상만 배치
  configs/             per-video 라인+ROI JSON
  output/              렌더링 결과 (.mp4)
  models/
    detection/         Stage-1 YOLO (.pt). DETECTORS 레지스트리 항목들의 저장지.
                       경로가 str (예: "yolo11l.pt") 이면 ultralytics 가 자동 다운로드
                       → resolve_model_path() 가 감지해 이 폴더로 이동시킴.
    classification/    Stage-2 세분 분류 모델. CLASSIFIERS 레지스트리 항목들의 저장지
                       (현재 ACTIVE_CLASSIFIER=None 으로 런타임 비활성).
    info-cls_*.yaml    각 모델의 클래스 공간을 기술하는 YAML (names: list | dict).
                       가중치의 model.names 가 아닌 이 YAML 이 Single source of truth.
  ReID 는 사용하지 않음 (BoTSORT with_reid=False 고정).

═══════════════════════════════════════════════════════════
  Counting Mode (COUNTING_MODE)
═══════════════════════════════════════════════════════════

  "two_line"    — 서로 다른 두 라인 교차 시 direction 기반 카운트
                  LANE_MAP = {"0->1": "Lane 1", ...} 으로 차선 라벨 매핑
                  HUD: 차선별 × 차종별 집계
  "single_line" — 한 라인만 지나도 즉시 카운트 (라인별 독립 집계)
                  SINGLE_LINE_LABELS = {0: "Line 0", ...} 로 라인 라벨 매핑
                  HUD: 라인별 고유 track 수 + 차종별 전체 합산
  두 모드는 상호배타. 드로잉 페이즈 최소 라인 개수도 모드에 따라 분기
  (two_line ≥ 2, single_line ≥ 1).

═══════════════════════════════════════════════════════════
  Model Registry (DETECTORS / CLASSIFIERS)
═══════════════════════════════════════════════════════════

  DETECTORS 레지스트리가 (가중치 경로, info YAML, 허용 클래스 이름) 을
  한 자리에 결합. ACTIVE_DETECTORS 리스트로 실행할 detector 키들을 선택.

  ACTIVE_DETECTORS 길이 >= 2 이면 자동 horizontal concat 비교 모드:
    - 각 패널이 독립된 tracker/counter/renderer/HUD 를 가짐
    - 출력 파일명에 각 detector tag 가 "-vs-" 로 연결됨

  각 detector 로딩 시:
    1) path 로부터 YOLO 로드. str 파일명이면 DETECTION_DIR 기준으로 resolve,
       없으면 ultralytics 자동 다운로드 → DETECTION_DIR 로 이동.
    2) info YAML 에서 {id: name} label_mapping 로드 (model.names 대체).
    3) allowed_names 가 주어졌다면 YAML 에서 id 를 역탐색해 필터링.
       (예: base YOLO 의 COCO 80 → car/bus/truck 만 허용 → {2:'car', 5:'bus', 7:'truck'})

  CLASSIFIERS + ACTIVE_CLASSIFIER — Stage-2 timm 분류기.
    - load_classifier_bundle() 가 가중치 + info YAML 로 모델을 로드.
    - cls id 는 CLASSIFIER_STAGE2_ID_OFFSET(기본 256) + 분류기 예측 인덱스.
    - apply_when: "after_tracking"(권장 기본) 또는 "after_detection".
    - 차량 계열 Stage-1 이름(car/bus/truck/…)에만 재분류 (apply_to_names 로 명시 가능).
    - DETECTORS[*]['stage2']=True 인 패널에만 적용(concat 시 한쪽만 12종 덧씌우기 가능).
  ACTIVE_CLASSIFIER=None 이면 Stage-1(YOLO) 라벨만 사용.

═══════════════════════════════════════════════════════════
  Lines (Counting Lines)
═══════════════════════════════════════════════════════════

  드로잉 페이즈에서 첫 프레임을 열어 마우스로 라인을 그림. id는 그리는 순서대로
  0, 1, 2, ... 자동 부여. two_line 모드에서는 0번이 기준선(모든 direction 의
  in-line) 역할이며, 1~4가 LANE_MAP 차선에 대응.

  Interactive Drawing Controls (draw_lines_interactive):
    Left click x 2   선분 1개 (시작점 → 끝점)
    Right click      대기 중 시작점 / 마지막 완성 선분 Undo
    R                전체 초기화
    C                이전 config 라인 비우기
    Enter / Space    확정 (two_line ≥ 2 · 권장 5, single_line ≥ 1)
    ESC              이 영상 건너뛰기

  라인 색은 전역 LINE_COLOR_RGB (기본 초록) 로 통일. 드로잉 시에는 얇은 실선,
  시각화 시에는 LINE_VIZ_THICKNESS 로 두껍게 그리고 LINE_VIZ_ALPHA 로 반투명.
  라인 중점에는 "해당 라인을 통과한 고유 track 수" 를 표시 (라인 ID 는 표시 안 함).

═══════════════════════════════════════════════════════════
  ROI (Region of Interest)
═══════════════════════════════════════════════════════════

  ROI_MODE (드로잉 페이즈에서 ROI를 결정하는 방식):
    "draw"  — 첫 프레임에서 다각형 ROI를 마우스로 직접 그림 (편집 모드 지원)
    "auto"  — 카운팅 라인 bbox + ROI_PADDING 으로 자동 사각형 ROI 생성

  OVERWRITE_EXISTING_CONFIG (기존 config 재실행 시 동작):
    "skip"  — 건너뛰기 (재사용)
    "edit"  — 이전 라인/ROI를 편집 가능 상태로 로드 (기본)
    "fresh" — 무조건 처음부터 새로 그리기

  Interactive Drawing Controls (draw_roi_interactive):
    Left click       Add vertex
    Right click      Undo last vertex
    R                Reset all vertices
    C                Clear loaded ROI (start fresh)
    Enter / Space    Confirm ROI (requires 3+ points)
    ESC              Skip ROI (use full frame)

  Visual Feedback (draw mode):
    - Counting lines shown as reference overlay
    - Real-time mouse tracking line with closing path preview
    - Numbered vertices (first point highlighted in green)
    - Semi-transparent fill preview (3+ points)
    - Status bar showing point count and loaded ROI state

  ROI Rendering (ROI_SHOW_BOUNDARY):
    - White semi-transparent boundary line (alpha 0.4)
    - Dark overlay on outside region (alpha 0.25)

  ROI Behavior:
    - Inside:  full opacity (ROI_INSIDE_ALPHA)
    - Outside: dimmed opacity (ROI_OUTSIDE_ALPHA)
    - Labels shown only inside ROI (ROI_SHOW_LABEL=False)
    - Point-in-polygon test via cv2.pointPolygonTest
    - Test point = get_center_point() (class-weighted, same as counting)

═══════════════════════════════════════════════════════════
  BBOX / Label / Trajectory
═══════════════════════════════════════════════════════════

  Color Mode (COLOR_MODE):
    "unified"      — 단일 색상 (#1EA7FF bbox, #00E5FF trajectory)
    "class"        — 세분 차종별 색상 (CLASS_COLOR_MAP, truck_s/m/x · bus_s/m 별도)
    "class_simple" — car / bus / truck 3색만 사용 (CLASS_SIMPLE_COLORS_RGB);
                     truck_*, bus_* 는 대분류로 정규화되어 단일 색상에 매핑됨
  HIDDEN_CLASSES 에 포함된 클래스(기본: bike, person)는 bbox/trajectory/label
  모두 렌더링에서 완전히 제외된다.
  SHOW_TRACK_ID=False 이면 라벨에 track id 를 붙이지 않고 차종명만 표시.

  BBOX (BBOX_STYLE):
    "corner"   — Thin full box + thick corner indicators (recommended)
    "default"  — Glow + rounded corners + corner indicators
    Corner length = bbox width * BBOX_CORNER_RATIO (25%)
    Line width: box 1.0px, corners 1.5px

  Label:
    - Language: LABEL_LANG ("en" / "ko")
    - Displayed as pill-shaped badge above bbox
    - ROI_SHOW_LABEL controls outside-ROI visibility

  Trajectory:
    - Width: TRAJECTORY_WIDTH (2px)
    - TRAJECTORY_FADE: older segments fade out gradually
    - Based on get_center_point() (class-weighted center)

"""
import sys
import json
import re
import time
import subprocess
import os
import cv2
import numpy as np
import skia
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm

# 프로젝트 루트를 sys.path에 추가 (boxmot 등 로컬 패키지 import용)
# _PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
# if _PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, _PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════
# 글로벌 설정
# ═══════════════════════════════════════════════════════════════

# ── 입출력 경로 ───────────────────────────────────────────────
#   VIDEO_PATHS: 명시적인 영상 파일 리스트 (우선 사용). None/빈 리스트면 VIDEO_DIR 사용
#   VIDEO_DIR + VIDEO_GLOB: 디렉토리에서 패턴 매칭으로 수집
#   단일 영상 처리 시에는 VIDEO_PATHS=["input/sample.mp4"] 식으로 길이 1 리스트 사용
# VIDEO_PATHS: "list[str] | None" = ["input/sample.mp4"]
VIDEO_PATHS: "list[str] | None" = [
    # "input/창원분기점_20251221130000_20251222110433.avi",
    # "input/목리교_20251203080000_20251203112758.avi",
    # "input/일직_20251221130001_20251222111727.avi",
    # "input/금토분기점1.mp4",
    # "input/청주분기점.mp4",
    # "input/중부2터널_20251203080000_20251203111458.avi",
    "input/궁평1교_20251203080000_20251203112024.avi",
    # "input/진천터널2_20251203080000_20251203103657.avi"
    ]
VIDEO_DIR:   "str | None"       = "input"
VIDEO_GLOB:  str                = "*.avi"

# ── 디렉토리 레이아웃 ─────────────────────────────────────────
#   input/       : 원본 영상
#   configs/     : per-video 라인+ROI (Single Source of Truth)
#   output/      : 렌더링 결과 (.mp4)
#   models/      : 모든 모델 아티팩트
#     detection/      : Stage-1 YOLO 계열 (.pt)
#     classification/ : Stage-2 세분 분류 (향후 확장용; 현재 미사용)
OUTPUT_DIR         = "output"
CONFIG_DIR         = "configs"   # per-video 통합 config(ROI+lines) 저장 위치. 유일한 진실의 원천(Single Root)
MODELS_DIR         = Path("models")
DETECTION_DIR      = MODELS_DIR / "detection"
CLASSIFICATION_DIR = MODELS_DIR / "classification"

# ── Model Registry ────────────────────────────────────────────
#   DETECTORS / CLASSIFIERS : 모델 가중치(.pt) ↔ 클래스 info YAML 을 한 자리에 결합.
#   ACTIVE_DETECTORS       : 실행 시 활성화할 detector 키 리스트.
#                            길이 >= 2 이면 horizontal concat 비교 모드.
#   ACTIVE_CLASSIFIER      : 2-stage 재분류를 수행할 classifier 키. None 이면 비활성.
#   새 모델 추가 시 이 레지스트리에 entry 만 더하면 됨.
#
#   각 detector entry 필드:
#     path          : 가중치 경로(Path) 또는 파일명(str).
#                     str 이고 DETECTION_DIR 에 없으면 ultralytics 자동 다운로드
#                     → resolve_model_path() 가 DETECTION_DIR 로 이동.
#     info          : 클래스 info YAML (Single source of truth for label_mapping)
#     allowed_names : None = 전체 사용 / list[str] = 해당 이름만 허용 (YAML 에서 id 역탐색)
#     stage2        : ACTIVE_CLASSIFIER 가 지정된 경우, 이 detector 패널에만 Stage-2 분류를
#                     적용할지 여부 (concat 시 한쪽은 YOLO만, 한쪽만 12종 덧씌우기 등).
#
#   각 classifier entry 필드:
#     path, info    : 위와 동일
#     input_size    : 입력 이미지 크기 (px)
#     min_bbox      : 이 값보다 작은 bbox 는 재분류 skip (px)
#     apply_when    : "after_tracking" | "after_detection"
DETECTORS = {
    # 커스텀 8클래스 YOLO (카테고리/이름: info-cls_8.yaml)
    "custom": {
        "path"         : DETECTION_DIR / "YOLOv11_l-cat_8-size_960.pt",
        "info"         : MODELS_DIR / "info-cls_8.yaml",
        "allowed_names": None,
        "stage2"       : False,  # True 로 두고 ACTIVE_CLASSIFIER 지정 시 이 패널만 12종 분류
        "display_name" : "custom model",   # 비교 모드에서 HUD/패널 부제로 표시
    },
    # Ultralytics yolo11l COCO, car/bus/truck 만 사용 (info-cls_80 역탐색)
    "base": {
        "path"         : "yolo11s.pt",
        "info"         : MODELS_DIR / "info-cls_80-coco2017.yaml",
        "allowed_names": ["car", "bus", "truck"],
        "stage2"       : False,  # 기본 모델 패널은 보통 Stage-2 없이 YOLO 라벨만
        "display_name" : "base model",
    },
}

#   transform_style:
#     "raw"         - Resize((sz,sz)) + ToTensor 만 (Normalize 없음).
#                     12종 분류기 (models__/classification/vehicle_classification.py
#                     + exports/export_classification.py) 가 이 방식으로 학습됨.
#     "resize_norm" - Resize((sz,sz)) + ToTensor + Normalize(mean,std).
#                     비율 무시 강제 정사각형 resize 후 ImageNet normalize 등 적용.
#                     8종 분류기 (/mnt/storage/admin_storage/ai_model_training/
#                     classification) 가 이 방식으로 학습됨 (mean/std 는
#                     augmentation_config.yaml 의 normalize 값과 정확히 일치시킴).
#                     normalize_mean/normalize_std 옵션으로 mean/std 지정 (기본:
#                     ImageNet).
#     "timm"        - timm.data.create_transform 표준 (Resize→CenterCrop→ToTensor
#                     →Normalize). timm 표준 fine-tune 모델에만 쓸 것. mobilenetv3
#                     같은 모델은 timm default 의 mean/std 가 (0,1) 로 잡혀 있어
#                     Normalize 가 사실상 안 되고 CenterCrop 으로 crop 도 변형되어
#                     학습-추론 mismatch 가 심하게 일어난다 → 가능하면 사용 금지.
#   normalize_mean / normalize_std:
#     "resize_norm" 시 사용할 정규화 mean/std. 학습 코드의 normalize 값과 정확히
#     일치시켜야 함. 기본값은 ImageNet.
#   bbox_padding:
#     crop 전 bbox 양옆을 이만큼의 비율로 확장 (0=학습과 동일, 정확). 학습 시
#     padding 없이 crop 이미지를 직접 사용한 모델에는 0 권장.
CLASSIFIERS = {
    "vehicle_subtype_12": {
        "path"          : CLASSIFICATION_DIR / "tf_efficientnet_b3_img300_bs32_lr0.004_SGD_best_f1_0.747_epoch7.pt",
        "info"          : MODELS_DIR / "info-cls_12.yaml",
        "timm_model"    : "tf_efficientnet_b3_ns",  # 체크포인트와 동일 아키텍처
        "input_size"    : 300,
        "min_bbox"      : 32,
        "apply_when"    : "after_tracking",       # "after_tracking" | "after_detection"
        # None 이면: HIDDEN 이 아니고 normalize_class 가 car/bus/truck 인 검출에만 재분류
        "apply_to_names": None,
        # 학습 코드와 동일하게 Resize+ToTensor 만 (Normalize 없음).
        "transform_style": "raw",
        "bbox_padding"  : 0.0,
    },
    # 커스텀 YOLO(8종) 의 분류 정확도 보강용 mobilenetv3 stage-2 분류기.
    # 클래스 셋이 동일하므로 Stage-1 라벨을 그대로 덮어쓴다 (offset id 로 구분).
    "vehicle_subtype_8": {
        "path"          : CLASSIFICATION_DIR / "mobilenetv3_large_100_epoch112_f10.9155.pth",
        "info"          : MODELS_DIR / "info-cls_8.yaml",
        "timm_model"    : "mobilenetv3_large_100.miil_in21k_ft_in1k",  # 체크포인트 config 와 동일
        "input_size"    : 224,                    # mobilenetv3_large_100 default
        "min_bbox"      : 24,                     # 8종 모델은 작은 bbox 도 분류 가능
        "apply_when"    : "after_tracking",
        # None: HIDDEN(person/bike) 제외하고 normalize_class 결과가 car/bus/truck 인 객체만 재분류
        "apply_to_names": None,
        # /mnt/storage/admin_storage/ai_model_training/classification 의 학습 코드와
        # 동일: Resize((224,224)) 비율-무시 강제 + ToTensor + ImageNet Normalize.
        # mean/std 는 augmentation_config.yaml 의 global_settings.normalize 값.
        "transform_style": "resize_norm",
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std" : [0.229, 0.224, 0.225],
        "bbox_padding"  : 0.0,                    # 학습 데이터셋이 이미 crop 된 상태
    },
}

# Stage-1 cls id 와 충돌하지 않도록 Stage-2 에 할당하는 id 오프셋 (COCO 최대 79, custom 최대 7)
CLASSIFIER_STAGE2_ID_OFFSET = 256

# ── Stage-2 분류 안정화 ───────────────────────────────────────
#   매 프레임 같은 차량을 독립적으로 분류하면 분류기 출력이 살짝만 흔들려도
#   라벨이 깜박입니다 (예: bus_s ↔ car ↔ bus_s). 다음 옵션들이 그 jitter 를 제거.
#
#   STAGE2_SMOOTH_ENABLED:
#     True  - track id 별 softmax 확률을 EMA 로 누적해 argmax (강력 추천)
#     False - 기존 동작(매 프레임 argmax 단발)
#   STAGE2_SMOOTH_ALPHA:
#     EMA 갱신 강도. 0~1, 작을수록 안정적/느림, 클수록 반응적/jittery.
#     0.30~0.40 권장. 0.35 = 직전 평균 65% + 새 관측 35%.
#   STAGE2_MIN_TOP1_PROB:
#     EMA 갱신을 하더라도 분류기 자체 top-1 확률이 이 값 미만이면
#     해당 프레임 관측은 무시(낮은 신뢰도 노이즈 차단). 0.0 = 비활성.
#   STAGE2_MIN_MARGIN:
#     top1 - top2 간 확률 차이가 이 값 미만이면 그 프레임 관측은 무시.
#     모호한(50:50) 프레임을 제거해 jitter 의 근본 원인을 차단.
#   STAGE2_BBOX_PADDING:
#     Crop 시 bbox 양쪽으로 추가할 마진 비율 (각 변 길이의 비율).
#     0.08 = 좌우/상하 8% 씩 확장. 차종 분류기는 약간의 background 가 있을 때
#     보통 더 정확. 0.0 이면 기존(타이트 bbox).
#   STAGE2_BATCH_INFER:
#     True - 한 프레임 안 모든 검출을 모아 1회 forward (CPU 에서도 큰 폭 빠름)
#     False - 검출별로 한 번씩 forward (기존 동작, 디버그용)
#   STAGE2_SMOOTH_TTL_SEC:
#     마지막 관측 후 이 시간(초) 가 지나면 해당 track 의 EMA 상태를 폐기.
#     같은 track id 가 재할당돼도 과거 클래스 잔재가 남지 않게 함.
STAGE2_SMOOTH_ENABLED  = True
STAGE2_SMOOTH_ALPHA    = 0.35
STAGE2_MIN_TOP1_PROB   = 0.45
STAGE2_MIN_MARGIN      = 0.08
STAGE2_BBOX_PADDING    = 0.08
STAGE2_BATCH_INFER     = True
STAGE2_SMOOTH_TTL_SEC  = 8.0

# ── Stage-2 trust mode (close-up catch-up 가속) ────────────────
#   문제 케이스: 차량이 멀리 있을 때 분류기가 confidently 틀려 EMA 가
#   잘못된 클래스로 잠긴 뒤, 가까워져서 옳게 분류되기 시작해도 EMA 가
#   풀리는데 몇 프레임 lag 가 걸림. 그 lag 때문에 라인-직전에서 잘못
#   카운트될 위험.
#
#   Trust mode: bbox >= STAGE2_TRUST_BBOX_PX 그리고 top1 >= STAGE2_TRUST_TOP1
#   인 "고품질" 관측을 만나면, 그 프레임만 **base alpha 대신 더 큰 alpha
#   (STAGE2_TRUST_ALPHA)** 로 EMA 를 업데이트해 catch-up 을 가속한다.
#   이로써 잠긴 잘못된 클래스에서 더 빠르게 빠져나옴.
#
#   STAGE2_TRUST_BBOX_PX:
#     bbox 의 min(w,h) 가 이 값 이상이면 close-up 으로 간주.
#     시뮬레이션상 영상 해상도 대비 70~100 px 권장.
#   STAGE2_TRUST_TOP1:
#     trust mode 진입 추가 조건. top1 확률이 이 값 미만이면 trust 보류.
#     모호한 close-up 은 base alpha 로만 부드럽게 갱신.
#   STAGE2_TRUST_ALPHA:
#     trust mode 의 alpha. 0.6~0.85 권장. 1.0 = 완전 덮어쓰기 (smoothing 무력화).
#     0.0 또는 STAGE2_TRUST_BBOX_PX <= 0 으로 두면 trust mode 자체 비활성.
STAGE2_TRUST_BBOX_PX   = 80
STAGE2_TRUST_TOP1      = 0.55
STAGE2_TRUST_ALPHA     = 0.80

# ── Trust bbox 임계의 video-level adaptive 자동 조정 ────────────
#   STAGE2_TRUST_BBOX_PX 는 영상 해상도/촬영 거리에 따라 적정값이 다르다.
#   (고속도로 원경 → 작은 bbox, 도심 근접 → 큰 bbox.)
#   이를 매번 수동으로 튜닝하지 않도록, 영상 안에서 분류 대상으로 들어온
#   bbox 들의 min(w,h) 분포로부터 percentile 을 실시간 계산해 임계로 사용.
#
#   STAGE2_TRUST_BBOX_ADAPTIVE:
#     True  → percentile 기반 자동 조정 (워밍업 동안은 fixed 값 사용)
#     False → 위의 STAGE2_TRUST_BBOX_PX 를 그대로 고정 사용
#   STAGE2_TRUST_BBOX_PCT:
#     상위 X% 가 trust 대상이 되도록 하는 percentile.
#     75 → 상위 25% (close-up) 만 trust 대상. 60~85 권장.
#     높을수록 trust 가 더 보수적 (catch-up 덜 적극적).
#   STAGE2_TRUST_BBOX_WARMUP:
#     percentile 추정에 필요한 최소 표본 수. 미만일 동안 fixed 값 사용.
#   STAGE2_TRUST_BBOX_FLOOR:
#     adaptive percentile 결과의 하한 (px). 영상 전반 bbox 가 너무 작아도
#     이보다 작은 임계는 사용하지 않음 (저품질 frame 까지 trust 되는 것 방어).
#   STAGE2_TRUST_BBOX_RECOMPUTE_EVERY:
#     성능 위해 매 N 개 새 표본마다 한 번씩만 percentile 재계산.
STAGE2_TRUST_BBOX_ADAPTIVE        = True
STAGE2_TRUST_BBOX_PCT             = 75.0
STAGE2_TRUST_BBOX_WARMUP          = 50
STAGE2_TRUST_BBOX_FLOOR           = 40
STAGE2_TRUST_BBOX_RECOMPUTE_EVERY = 30

# ── Tracker gap-filling (lost track 의 Kalman 예측 bbox 사용) ────
#   YOLO 가 한두 프레임 detection 을 놓쳐도 BoTSORT 는 track_buffer 동안
#   해당 track 을 lost 상태로 보유 + 매 프레임 Kalman 예측을 한 step 진행.
#   detection 매칭이 안 된 lost track 의 예측 bbox 를 출력에 합쳐, 시각화의
#   bbox 깜박임을 없애고 카운팅에서도 gap 동안 라인 통과를 검출 가능하게 함.
#
#   주의: 예측 bbox 는 위치/크기 drift 가 있을 수 있으므로 Stage-2 분류에는
#   사용하지 않음 (apply_stage2_classification 에서 자동 skip). EMA 와
#   adaptive trust percentile 표본도 오염 방지를 위해 갱신 skip.
#
#   TRACKER_USE_LOST_PREDICTION:
#     True  → lost track 예측 bbox 로 detection gap 메우기 (시각적 연속성 ↑)
#     False → 기존 동작 (gap 동안 bbox 사라짐)
#   TRACKER_PREDICT_MAX_GAP:
#     이 값 이상 detection 매칭이 안 된 lost track 은 예측 신뢰 안 함 (px drift 누적).
#     일반적으로 3~7 권장. BoTSORT 의 track_buffer (45) 보다 훨씬 작아야 함.
TRACKER_USE_LOST_PREDICTION = True
TRACKER_PREDICT_MAX_GAP     = 5

# 실행 시 활성화할 detector 키들. 길이 >= 2 이면 비교(concat) 모드.
ACTIVE_DETECTORS: "list[str]" = ["base", "custom"]
DETECTORS["base"]["stage2"] = False
DETECTORS["custom"]["stage2"] = True   # 커스텀 패널은 mobilenetv3 8종 stage-2 로 라벨 보정

# 2-stage 재분류 classifier 키. None 이면 Stage-1(YOLO) 라벨만 사용.
# 실제 추론은 DETECTORS[*]["stage2"]=True 인 패널이 하나라도 있을 때만 로드·적용된다.
#   "vehicle_subtype_8"  : mobilenetv3_large_100 (8종, custom YOLO 의 분류 보정용)
#   "vehicle_subtype_12" : tf_efficientnet_b3_ns (12종, 더 세분화된 차종)
# ACTIVE_CLASSIFIER: "str | None" = "vehicle_subtype_8"
ACTIVE_CLASSIFIER: "str | None" = "vehicle_subtype_12"

# ── 실행 프리셋 예시 (주석 해제 후 ACTIVE_DETECTORS 와 맞춤 사용) ─────────
# (1) base vs custom, Stage-1 만 (concat)
#     ACTIVE_DETECTORS = ["base", "custom"]
#     DETECTORS["base"]["stage2"] = False
#     DETECTORS["custom"]["stage2"] = False
#     ACTIVE_CLASSIFIER = None
#
# (2) base vs custom, 오른쪽(custom)만 YOLO + mobilenetv3(8종) 보정 (현재 기본)
#     ACTIVE_DETECTORS = ["base", "custom"]
#     DETECTORS["base"]["stage2"] = False
#     DETECTORS["custom"]["stage2"] = True
#     ACTIVE_CLASSIFIER = "vehicle_subtype_8"
#
# (3) base vs custom, 오른쪽(custom)만 YOLO + 12종 세분화 분류 덧씌우기
#     ACTIVE_DETECTORS = ["base", "custom"]
#     DETECTORS["base"]["stage2"] = False
#     DETECTORS["custom"]["stage2"] = True
#     ACTIVE_CLASSIFIER = "vehicle_subtype_12"

COMPARE_LAYOUT = "horizontal"   # 현재는 horizontal 만 지원

# ── 실행 환경 프리셋 (한 줄 토글로 PC ↔ 서버 전환) ──────────────
#   사용법: 아래 두 줄을 동시에 선택하고 에디터의 "주석 토글"(Ctrl+/)을 한 번 누르면
#          활성/비활성이 그대로 스왑된다. 항상 정확히 한 줄만 활성 상태여야 한다.
#
#   RUN_MODE 의미:
#     "draw_only"        그리기 페이즈만 (configs/*.json 생성하고 종료)
#     "process_only"     처리 페이즈만 (configs/*.json 이 이미 있어야 함)
#     "draw_and_process" 둘 다 (기본 PC 워크플로)
#
#   HEADLESS 의미:
#     "auto" 자동감지(Linux & DISPLAY/WAYLAND_DISPLAY 둘 다 없으면 헤드리스,
#            환경변수 HEADLESS=1/true/yes 우선)
#     True   강제 헤드리스(=imshow 호출 경로 차단)
#     False  강제 GUI 모드
#
#   대상 헤드리스 환경: SSH+X11 미설정 Linux 서버, DISPLAY 없는 Docker 등.
#   (opencv-python 그대로 사용 OK. VideoCapture/imwrite/FFmpeg 다 동작,
#    drawing_phase 의 imshow 만 차단됨. processing_phase 에는 GUI API 0개.)
RUN_MODE, HEADLESS = "draw_and_process", "auto"   # [PC]   그리기 + 처리
# RUN_MODE, HEADLESS = "process_only",     True   # [서버] imshow 차단, 처리만

# ── 병렬 처리 설정 (영상 단위 멀티프로세싱) ────────────────────
#   processing_phase 에서 여러 영상을 동시에 처리한다.
#   각 영상은 별도 OS 프로세스에서 자체 YOLO/classifier 인스턴스를 가지고
#   돌아가므로 GIL 영향 없이 N개의 frame loop 가 진정한 의미로 병렬 실행됨.
#
#   MAX_PARALLEL_VIDEOS:
#     "auto" : min(영상 수, max(1, 물리코어/THREADS_PER_WORKER_HINT)) 로 자동 결정.
#              이 시스템(48 phys cores)에서 대략 6 workers 가 선택됨.
#     1      : 병렬 비활성 (기존 직렬 동작과 동일, 디버깅용)
#     int    : 명시적 worker 수 지정.
#   THREADS_PER_WORKER:
#     "auto" : 물리 코어 수 ÷ MAX_PARALLEL_VIDEOS.
#     int    : 각 worker 내 PyTorch / OpenCV 의 intra-op 스레드 수.
#              너무 크면 worker 끼리 코어를 두고 다투어(oversubscription) 오히려 느려짐.
#   THREADS_PER_WORKER_HINT:
#     MAX_PARALLEL_VIDEOS == "auto" 모드에서 worker 1개당 할당하길 원하는
#     스레드 개수 힌트. YOLO CPU 추론은 보통 8~16 스레드에서 효율 정점.
MAX_PARALLEL_VIDEOS    : "int | str" = "auto"
THREADS_PER_WORKER     : "int | str" = "auto"
THREADS_PER_WORKER_HINT: int         = 8

# ── 기존 config 가 있을 때 그리기 동작 ────────────────────────
#   "skip"  : 건너뛰기 (재사용)
#   "edit"  : 이전 라인/ROI를 편집 가능 상태로 로드 (기본)
#   "fresh" : 무조건 처음부터 새로 그리기
OVERWRITE_EXISTING_CONFIG = "edit"

# ── 처리 설정 ─────────────────────────────────────────────────
MAX_SECONDS      = 60       # 처리할 최대 시간(초). None이면 전체 영상
USE_TRACKER_BBOX = True     # True: tracker bbox / False: YOLO 원본 bbox

# ── 카운팅 모드 ───────────────────────────────────────────────
#   "two_line"   : 한 객체가 서로 다른 두 라인을 교차했을 때만 카운팅
#                 (direction 은 "in_idx->out_idx", LANE_MAP 으로 차선명 매핑)
#   "single_line": 한 객체가 하나의 라인을 지나면 즉시 카운팅
#                 (라인별 독립 집계, SINGLE_LINE_LABELS 로 라벨 매핑)
#   두 모드는 상호배타. 전환 시 LANE_MAP / SINGLE_LINE_LABELS 모두 정의돼 있어야 함.
COUNTING_MODE = "single_line"

# ── 색상 체계 ───────────────────────────────────────────────
#   "unified"      : 단일 색상 (업계 표준 - DJI, GOODVISION 등)
#   "class"        : 차종 세분 색상 (CLASS_COLOR_MAP - truck_s/m/x, bus_s/m 별도)
#   "class_simple" : car / bus / truck 3색만 사용 (세부 truck-*, bus-* 는 대분류로 매핑)
COLOR_MODE           = "class_simple"
UNIFIED_BBOX_COLOR   = (30, 167, 255)    # #1EA7FF — bbox/코너/라벨 색상
UNIFIED_TRAJ_COLOR   = (0, 229, 255)     # #00E5FF — 궤적 색상 (bbox보다 밝게)

# class_simple 모드 색상 (RGB) — bbox 와 trajectory 공용
CLASS_SIMPLE_COLORS_RGB = {
    "car":   (77, 150, 255),   # 파란 계열
    "bus":   (0, 200, 140),    # 청록 계열
    "truck": (255, 170, 60),   # 주황 계열
}

# 완전히 렌더링에서 제외할 클래스 (bbox/label/traj 모두 그리지 않음)
# "car"/"bus"/"truck" 대분류 혹은 "truck_s" 등 원본 클래스명 모두 허용.
HIDDEN_CLASSES = {"bike", "person"}

# ── ROI (관심 영역) ─────────────────────────────────────────
#   ROI 내부: 선명하게 / ROI 외부: 흐리게 표시
#   드로잉 페이즈에서 ROI를 결정하는 방식 (결과는 configs/{stem}.json에 저장됨):
#     "draw" : 첫 프레임에서 다각형 ROI를 마우스로 직접 그림 (편집 모드 지원)
#     "auto" : 카운팅 라인 bbox + ROI_PADDING 으로 자동 사각형 ROI 생성
ROI_ENABLED          = True
ROI_MODE             = "draw"  # "draw" / "auto"
ROI_PADDING          = 50     # auto 모드: 카운팅 라인 bbox 기준 여백 (px)
# ROI 외부 객체 처리 정책:
#   "hide" : 외부 객체를 시각화/카운팅 모두에서 완전히 제외 (필터로만 사용)
#   "fade" : 외부도 그리되 흐리게 표시 (ROI_OUTSIDE_ALPHA, ROI_DIM_ALPHA 사용)
ROI_OUTSIDE_BEHAVIOR = "hide"
# 아래는 모두 ROI_OUTSIDE_BEHAVIOR == "fade" 일 때만 의미 있음
ROI_INSIDE_ALPHA     = 1.0    # ROI 내부 객체 불투명도
ROI_OUTSIDE_ALPHA    = 0.5    # ROI 외부 객체 불투명도
ROI_SHOW_LABEL       = True   # ROI 외부에서도 라벨 표시 여부
ROI_SHOW_BOUNDARY    = False  # ROI 경계선을 영상에 표시할지 여부 ("hide" 모드에선 보통 OFF)
ROI_BOUNDARY_ALPHA   = 0.7    # ROI 경계선 불투명도
ROI_BOUNDARY_WIDTH   = 1.5    # ROI 경계선 두께 (px)
ROI_DIM_ALPHA        = 0.0    # ROI 외부 배경 어둡게 처리 강도 (0.0=없음)

# ── 바운딩 박스 ───────────────────────────────────────────────
#   "default" : 글로우 + 둥근 모서리 + 코너 인디케이터
#   "corner"  : 코너만 두껍게, 나머지 얇게, 글로우 없음, 직각
BBOX_STYLE           = "corner"
BBOX_STROKE_WIDTH    = 1.0    # 박스 선 두께 (corner 스타일 기준, default는 2.0 고정)
BBOX_CORNER_WIDTH    = 4    # 코너 인디케이터 최대 선 두께
BBOX_CORNER_LENGTH   = 35     # 코너 인디케이터 최대 길이 (px)
BBOX_CORNER_RATIO    = 0.25   # 코너 길이 = bbox width * ratio (20~30% 권장)
BBOX_CORNER_DYNAMIC  = False   # True: bbox 크기에 따라 코너 두께/길이 동적 스케일링
BBOX_CORNER_REF_SIZE = 120    # 동적 스케일링 시, 이 크기(px) 이상의 박스에서 최대 굵기/길이 적용
BBOX_CORNER_MIN_WIDTH = 1.0   # 동적 스케일링 시, 코너 인디케이터 최소 선 두께
BBOX_CORNER_MIN_LEN  = 6      # 동적 스케일링 시, 코너 인디케이터 최소 길이 (px)
BBOX_FILL_OVERLAY    = False  # 박스 내부 반투명 채우기 여부
BBOX_FILL_ALPHA      = 0.25   # 박스 내부 채우기 불투명도

# ── 궤적 (Trajectory) ────────────────────────────────────────
SHOW_TRAJECTORY           = True   # 궤적 표시 여부
TRAJECTORY_ALPHA          = 0.9    # 궤적 코어 불투명도
TRAJECTORY_WIDTH          = 2      # 궤적 코어 선 두께 (px)
TRAJECTORY_FADE           = True   # True: 오래된 궤적일수록 흐리게 (FADE)
TRAJECTORY_GLOW           = True   # True: 궤적 아래에 글로우 레이어 추가
TRAJECTORY_GLOW_WIDTH_FACTOR = 4.0 # 글로우 두께 = 코어 두께 × 이 값
TRAJECTORY_GLOW_ALPHA     = 0.25   # 글로우 레이어 최대 불투명도
TRAJECTORY_TAPER          = True   # True: 최근 점일수록 굵게, 오래될수록 가늘게
TRAJECTORY_TAPER_MIN_WIDTH = 0.5   # 테이퍼링 시 최소 선 두께 (px)

# ── 카운팅 기준점 (Center Point) ──────────────────────────────
SHOW_CENTER_POINT          = True   # 기준점 표시 여부
CENTER_POINT_RADIUS        = 2.0    # 기준점 반지름 (px)
CENTER_POINT_HEIGHT_OFFSET = 0.15   # bbox 높이 대비 추가 하단 이동량 (0.0=변경없음)

# ── 라벨 (Label) ────────────────────────────────────────────
#   "en" : 영문 (Car, Bus, Truck 등 — 더 작게 표현 가능)
#   "ko" : 한글 (승용차, 버스 등)
LABEL_LANG           = "en"
SHOW_TRACK_ID        = False  # True: 라벨에 "#tid " 접두어 붙임 / False: 차종명만
# GROUP_TRUCK / GROUP_BUS 는 *라벨 표시* 만 통합. 색상 단순화는 COLOR_MODE 가 담당.
#   False (권장)  : custom 8클래스/Stage-2 12클래스의 세부 라벨을 그대로 표시
#                  (예: Bus(S), Truck(M)). 모델 출력 정보 손실 없음.
#   True         : 모든 truck/bus 변형을 "Truck"/"Bus" 한 단어로 강제 통합.
#                  base COCO(car/bus/truck) 처럼 원래 통합돼 있는 모델에는 영향 없음.
GROUP_TRUCK          = False
GROUP_BUS            = False

# ── 카운팅 이벤트 이펙트 ──────────────────────────────────────
SHOW_LINE_HIT_EFFECT = False  # 카운팅 라인 자체 강조 애니메이션
SHOW_CROSS_MARKER    = True   # 궤적-카운팅라인 교차점 마커

# ── 카운팅 라인 시각화 ────────────────────────────────────────
# 드로잉 페이즈 / 시각화 페이즈 모두 아래 색으로 통일 (초록 고정).
LINE_COLOR_RGB        = (0, 200, 80)   # 초록 (RGB)
# 드로잉 페이즈 선 두께 (얇게 유지) — 1080p 기준값. 실제 두께는 프레임 해상도로 자동 스케일됨.
LINE_DRAW_THICKNESS   = 2
# 시각화(렌더링) 페이즈 선 두께 (굵게) — 1080p 기준값, VIZ_SCALE 로 자동 보정됨.
LINE_VIZ_THICKNESS    = 8.0
# 시각화 실선 반투명도 (0~1) — 높을수록 진함
LINE_VIZ_ALPHA        = 0.55
# 시각화 글로우 반투명도 (0~1) — 굵은 실선 아래 soft glow
LINE_VIZ_GLOW_ALPHA   = 0.22
# 라인 카운트 라벨 원 반지름 — 1080p 기준값, VIZ_SCALE 로 자동 보정됨.
LINE_COUNT_CIRCLE_RAD = 14
# 라인 카운트 라벨 원 배경 알파 (0~1) — 낮을수록 도로가 더 잘 보임.
LINE_COUNT_LABEL_ALPHA        = 0.30
# 라인 카운트 라벨 원 테두리 알파
LINE_COUNT_LABEL_BORDER_ALPHA = 0.45

# ── 좌상단 HUD (집계 패널) ────────────────────────────────────
#   "off"    : 표시 안 함
#   "totals" : 차종별 합계만 (라인/차선 분해 X) — 라인별 수치는 카운팅 라인 위에 이미 표시됨
#   "full"   : 라인/차선별 + 차종별 상세 (기존 동작)
HUD_MODE = "totals"

# ── 인터랙티브 드로잉 UI 자동 스케일 ──────────────────────────
#   끝점 마커 / 라벨 폰트 / 상태바 등 픽셀 단위 UI 요소는
#   min(frame_h, frame_w) / DRAW_UI_REF_DIM 비례로 자동 스케일.
#   영상이 720p/1080p/4K 어떤 해상도여도 시각적 비율이 같도록 함.
#   미세 조정이 필요하면 DRAW_UI_SCALE 만 0.7~1.3 사이에서 조정.
DRAW_UI_REF_DIM = 1080         # 기준 해상도 (이 값에서 1.0 배율)
DRAW_UI_SCALE   = 1.0          # 사용자 미세 조정 추가 배율
DRAW_UI_SCALE_MIN = 0.4        # 너무 작아지지 않도록 하한
DRAW_UI_SCALE_MAX = 1.6        # 너무 커지지 않도록 상한

# ── 시각화(렌더링) UI 자동 스케일 ─────────────────────────────
#   bbox 코너/궤적/라벨/카운팅 라인/HUD 등 모든 픽셀 단위 시각화 요소를
#   min(frame_h, frame_w) / VIZ_REF_DIM 비례로 자동 스케일.
#   1080p 기준의 상수값들이 720p/4K 등 다른 해상도에서도 시각적 비율을
#   동일하게 유지하도록 함. 미세 조정은 VIZ_SCALE 로.
VIZ_REF_DIM   = 1080
VIZ_SCALE     = 1.0
VIZ_SCALE_MIN = 0.5
VIZ_SCALE_MAX = 2.0

# 차선 매핑 (direction -> 차선명) — two_line 모드에서만 사용
LANE_MAP = {
    "0->1": "Lane 1",
    "0->2": "Lane 2",
    "0->3": "Lane 3",
    "0->4": "Lane 4",
}

# 라인별 라벨 — single_line 모드에서만 사용.
# 정의되지 않은 line_idx 는 "Line {idx}" 로 자동 fallback.
SINGLE_LINE_LABELS = {
    0: "Line 0",
    1: "Line 1",
    2: "Line 2",
    3: "Line 3",
    4: "Line 4",
}

# ── 컬러 팔레트 ──────────────────────────────────────────
PALETTE_HEX = [
    "#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF",
    "#C77DFF", "#FF9A3C", "#00C9A7", "#F72585",
]
PALETTE_RGB = []
for _h in PALETTE_HEX:
    _hh = _h.lstrip("#")
    r, g, b = int(_hh[0:2], 16), int(_hh[2:4], 16), int(_hh[4:6], 16)
    PALETTE_RGB.append((r, g, b))

# 차종별 고정 색상 (RGB). 키는 YAML (info-cls_8.yaml) 과 동일한 언더스코어 표기.
CLASS_COLOR_MAP = {
    "car":     (77, 150, 255),
    "bus_s":   (107, 203, 119),
    "bus_m":   (0, 201, 167),
    "truck_s": (255, 217, 61),
    "truck_m": (255, 154, 60),
    "truck_x": (255, 107, 107),
    "bike":    (199, 125, 255),
    "person":  (247, 37, 133),
}

# ──────────────────────────────────────────────────────────────
# 클래스 표시명 (간략) — 모든 모델의 클래스에 대해 짧고 박스 위에
# 얹기 좋은 라벨을 미리 지정.
#
# 각 dict 는 모든 활성 가능한 모델의 클래스를 한꺼번에 포함:
#   - base COCO 필터링 (car/bus/truck)
#   - custom 8-class      → models/info-cls_8.yaml
#   - Stage-2 12-class    → models/info-cls_12.yaml
#
# 해당 모델만 활성화돼 있으면 다른 키들은 단순히 사용되지 않음.
# 매핑이 없는 클래스는 raw 이름이 그대로 표시되므로,
# 새 모델/클래스 추가 시 여기에 한 줄씩 채워주는 패턴.
# ──────────────────────────────────────────────────────────────

# 차종 한글명 (간략)
CLASS_KOR = {
    # base COCO
    "car": "승용", "bus": "버스", "truck": "트럭",
    # custom 8-class
    "bus_s":   "버스S", "bus_m":   "버스M",
    "truck_s": "트럭S", "truck_m": "트럭M", "truck_x": "트럭L",
    "bike":    "이륜",  "person":  "보행",
    # Stage-2 12-class
    "truck_4W_FT": "트4FT", "truck_4W_ST": "트4ST",
    "truck_5W_FT": "트5FT", "truck_5W_ST": "트5ST",
    "truck_6W_ST": "트6ST",
    "truck_m_3W":  "트M3W", "truck_m_4W":  "트M4W", "truck_m_5W":  "트M5W",
    "truck_s_a":   "트Sa",  "truck_s_b":   "트Sb",
}

# 차종 영문명 (간략)
CLASS_EN = {
    # base COCO
    "car": "Car", "bus": "Bus", "truck": "Trk",
    # custom 8-class
    "bus_s":   "BusS", "bus_m":   "BusM",
    "truck_s": "TrkS", "truck_m": "TrkM", "truck_x": "TrkL",
    "bike":    "Bike", "person":  "Person",
    # Stage-2 12-class
    "truck_4W_FT": "T4FT", "truck_4W_ST": "T4ST",
    "truck_5W_FT": "T5FT", "truck_5W_ST": "T5ST",
    "truck_6W_ST": "T6ST",
    "truck_m_3W":  "Tm3W", "truck_m_4W":  "Tm4W", "truck_m_5W":  "Tm5W",
    "truck_s_a":   "Tsa",  "truck_s_b":   "Tsb",
}

# HUD 표시 순서. 등록되지 않은 클래스(Stage-2 등)는
# group_class_counts 가 보조 루프에서 알파벳 순으로 자동 추가.
CLASS_DISPLAY_ORDER = [
    "car", "bus", "bus_s", "bus_m",
    "truck", "truck_s", "truck_m", "truck_x",
    "bike", "person",
]


# ═══════════════════════════════════════════════════════════════
# 인터랙티브 드로잉 UI 스케일 헬퍼
# ═══════════════════════════════════════════════════════════════

def _drawing_ui_metrics(frame):
    """프레임 해상도 기준으로 인터랙티브 드로잉 UI 픽셀 메트릭을 계산.

    DRAW_UI_REF_DIM(기본 1080) 의 짧은 변에서 1.0 배율이 되도록 정규화.
    하한/상한으로 클램프된 뒤 DRAW_UI_SCALE (사용자 추가 배율) 곱.

    Returns
    -------
    dict — keys:
      scale         : 최종 적용 배율 (float)
      line_thick    : 라인/엣지 두께 (int >= 1)
      vertex_r      : 끝점 마커 반지름 (int >= 2)
      label_font_sc : 라인/포인트 라벨 폰트 스케일 (float)
      label_font_th : 라벨 폰트 두께 (int >= 1)
      label_pad     : 라벨 원 패딩 (int >= 2)
      bar_h         : 상단 상태바 높이 (int)
      bar_font_sc   : 상태바 폰트 스케일 (float)
      bar_font_th   : 상태바 폰트 두께 (int >= 1)
      bar_text_y    : 상태바 텍스트 baseline y 좌표 (int)
    """
    h, w = frame.shape[:2]
    auto = min(h, w) / float(DRAW_UI_REF_DIM)
    raw = auto * DRAW_UI_SCALE
    scale = max(DRAW_UI_SCALE_MIN, min(DRAW_UI_SCALE_MAX, raw))

    bar_h = max(20, int(round(36 * scale)))
    return {
        "scale"        : scale,
        "line_thick"   : max(1, int(round(LINE_DRAW_THICKNESS * scale))),
        "vertex_r"     : max(2, int(round(5 * scale))),
        "label_font_sc": max(0.35, 0.6 * scale),
        "label_font_th": max(1, int(round(2 * scale))),
        "label_pad"    : max(2, int(round(6 * scale))),
        "bar_h"        : bar_h,
        "bar_font_sc"  : max(0.35, 0.6 * scale),
        "bar_font_th"  : max(1, int(round(1 * scale))),
        "bar_text_y"   : max(14, int(round(bar_h * 0.66))),
    }


def _viz_scale_factor(w, h):
    """시각화(SkiaCountingRenderer) 용 단일 스케일 배율을 계산.

    min(h, w) / VIZ_REF_DIM 으로 자동 산출 후 VIZ_SCALE 곱, 상/하한 클램프.
    렌더러 init 에서 한 번만 계산하여 모든 픽셀 단위 시각화 요소에 곱함.
    """
    auto = min(h, w) / float(VIZ_REF_DIM)
    raw = auto * VIZ_SCALE
    return max(VIZ_SCALE_MIN, min(VIZ_SCALE_MAX, raw))


# ═══════════════════════════════════════════════════════════════
# ROI 인터랙티브 드로잉
# ═══════════════════════════════════════════════════════════════

def draw_roi_interactive(frame, counting_lines=None, prev_polygon=None):
    """
    첫 프레임에서 다각형 ROI를 마우스로 그리는 인터랙티브 툴.

    조작법:
      좌클릭       — 꼭짓점 추가
      우클릭       — 마지막 꼭짓점 취소 (Undo)
      Enter / Space — ROI 확정
      R            — 전체 초기화
      C            — 불러온 ROI 삭제 (새로 그리기)
      ESC          — ROI 없이 진행 (전체 프레임 사용)

    prev_polygon: 이전 ROI (np.array shape (N,2)) — 있으면 편집 가능 상태로 로드
    반환: np.array of shape (N, 2) — 다각형 꼭짓점 목록, 또는 None
    """
    WINDOW_NAME = "ROI Drawing  |  Left: add  |  Right: undo  |  Enter: confirm  |  R: reset  |  C: clear prev  |  ESC: skip"
    ACCENT      = (255, 167, 30)   # #1EA7FF in BGR
    ACCENT_DIM  = (180, 120, 20)
    FILL_COLOR  = (255, 167, 30)
    PREV_COLOR  = (100, 200, 100)  # 이전 ROI 표시 색상 (녹색)
    M = _drawing_ui_metrics(frame)
    VERTEX_RAD = M["vertex_r"]
    LINE_THICK = M["line_thick"]

    # 이전 ROI가 있으면 편집 가능 상태로 로드
    points = []
    if prev_polygon is not None:
        points = [tuple(pt) for pt in prev_polygon.tolist()]
    mouse_pos = [0, 0]
    confirmed = [False]
    skipped = [False]
    has_prev = [prev_polygon is not None and len(points) > 0]  # 이전 ROI 로드 여부

    base = frame.copy()

    # 카운팅 라인을 배경에 그려서 참조용으로 표시
    if counting_lines:
        for _lid, coords in counting_lines:
            for x1, y1, x2, y2 in coords:
                cv2.line(base, (x1, y1), (x2, y2), (0, 200, 255), 1, cv2.LINE_AA)

    def _render():
        canvas = base.copy()
        n = len(points)

        # 반투명 채우기 (폴리곤이 3점 이상일 때)
        if n >= 3:
            overlay = canvas.copy()
            pts_arr = np.array(points, dtype=np.int32)
            cv2.fillPoly(overlay, [pts_arr], FILL_COLOR)
            cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)

        # 확정된 변 (실선)
        for i in range(n - 1):
            cv2.line(canvas, points[i], points[i + 1], ACCENT, LINE_THICK, cv2.LINE_AA)

        # 마우스 추적 선 (점선 효과 — 얇은 선)
        if n > 0 and not confirmed[0]:
            cv2.line(canvas, points[-1], tuple(mouse_pos), ACCENT_DIM, 1, cv2.LINE_AA)
            # 닫히는 선 미리보기
            if n >= 2:
                cv2.line(canvas, tuple(mouse_pos), points[0], ACCENT_DIM, 1, cv2.LINE_AA)

        # 꼭짓점
        num_off = max(4, int(round(8 * M["scale"])))
        for i, pt in enumerate(points):
            color = (0, 255, 100) if i == 0 else ACCENT
            cv2.circle(canvas, pt, VERTEX_RAD, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, VERTEX_RAD + 1, (255, 255, 255), 1, cv2.LINE_AA)
            # 번호 표시
            cv2.putText(canvas, str(i + 1), (pt[0] + num_off, pt[1] - num_off),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.35, 0.45 * M["scale"]),
                        (255, 255, 255), max(1, int(round(M["scale"]))), cv2.LINE_AA)

        # 닫힌 폴리곤 (확정 후)
        if confirmed[0] and n >= 3:
            cv2.line(canvas, points[-1], points[0], ACCENT, LINE_THICK, cv2.LINE_AA)

        # 상태 표시 바
        bar_h = M["bar_h"]
        canvas[:bar_h, :] = cv2.addWeighted(canvas[:bar_h, :], 0.4, np.zeros_like(canvas[:bar_h, :]), 0.6, 0)
        status = f"Points: {n}"
        if has_prev[0]:
            status += "  [Loaded prev ROI - C: clear]"
        elif n < 3:
            status += "  (need at least 3)"
        cv2.putText(canvas, status, (10, M["bar_text_y"]),
                    cv2.FONT_HERSHEY_SIMPLEX, M["bar_font_sc"], (255, 255, 255),
                    M["bar_font_th"], cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, canvas)

    def _on_mouse(event, x, y, flags, param):
        mouse_pos[0], mouse_pos[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            _render()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if points:
                points.pop()
            _render()
        elif event == cv2.EVENT_MOUSEMOVE:
            _render()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, frame.shape[1], frame.shape[0])
    cv2.setMouseCallback(WINDOW_NAME, _on_mouse)
    _render()

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            skipped[0] = True
            break
        elif key in (13, 32):  # Enter / Space
            if len(points) >= 3:
                confirmed[0] = True
                _render()
                cv2.waitKey(500)  # 확정 결과를 잠시 표시
                break
        elif key in (ord('r'), ord('R')):
            points.clear()
            has_prev[0] = False
            _render()
        elif key in (ord('c'), ord('C')):
            # 불러온 ROI 삭제 (새로 그리기)
            points.clear()
            has_prev[0] = False
            _render()

    cv2.destroyWindow(WINDOW_NAME)

    if skipped[0] or len(points) < 3:
        return None
    return np.array(points, dtype=np.int32)


def draw_lines_interactive(frame, roi_polygon=None, prev_lines=None):
    """
    첫 프레임에서 카운팅 라인을 마우스로 그리는 인터랙티브 툴.
    id는 그리는 순서대로 0, 1, 2, ... 자동 부여. 첫 번째(id=0)가 기준선.

    조작법:
      좌클릭 1회      — 선분 시작점 지정
      좌클릭 2회      — 선분 끝점 지정 → 선분 확정, id 자동증가
      우클릭         — 대기중 시작점 취소 / 없으면 마지막 완성 선분 Undo
      Enter / Space  — 확정 (최소 2개 필요, 5개 미만이면 경고 후 재확인)
      R             — 전체 초기화
      C             — 불러온 이전 라인 비우기 (새로 그리기)
      ESC           — 이 영상 건너뛰기 (None 반환)

    roi_polygon: np.array shape (N,2) — 배경에 반투명으로 표시 (맞춰 그릴 때 참조)
    prev_lines : list[dict] polylines_lst 포맷 — 이전 config 라인 편집용 로드

    반환:
        list[dict] in polylines_lst 포맷 (skip 시 None)
        [{"id": 0, "num_lines": 1, "coords": [[x1, y1, x2, y2]]}, ...]
    """
    WINDOW_NAME = "Line Drawing  |  Left: add (2 clicks=1 line)  |  Right: undo  |  Enter: confirm  |  R: reset  |  C: clear prev  |  ESC: skip"
    ROI_FILL    = (255, 167, 30)      # 오렌지 반투명
    ROI_EDGE    = (255, 167, 30)
    M = _drawing_ui_metrics(frame)
    LINE_THICK  = M["line_thick"]
    ENDPOINT_R  = M["vertex_r"]

    # 라인 색은 전역적으로 초록 단일 (RGB → BGR)
    _lr, _lg, _lb = LINE_COLOR_RGB
    LINE_COLOR_BGR = (_lb, _lg, _lr)

    # 모드별 최소 라인 개수 / 권장 개수
    if COUNTING_MODE == "single_line":
        min_lines_required = 1
        rec_lines_required = 1
        rec_warn_msg = None
    else:
        min_lines_required = 2
        rec_lines_required = 5  # LANE_MAP 가 0=baseline + 1..4=lanes 가정
        rec_warn_msg = "LANE_MAP expects 5 (0=baseline + 1..4=lanes)"

    # 편집 가능한 상태 구조: lines = [((x1,y1), (x2,y2)), ...]  (id는 index)
    lines = []
    if prev_lines:
        for item in prev_lines:
            coords = item.get("coords", [])
            if coords:
                x1, y1, x2, y2 = coords[0]
                lines.append(((int(x1), int(y1)), (int(x2), int(y2))))

    pending_start = [None]             # 대기 중인 시작점 (좌클릭 1회 후)
    mouse_pos = [0, 0]
    confirmed = [False]
    skipped = [False]
    has_prev = [prev_lines is not None and len(lines) > 0]

    base = frame.copy()

    # 배경에 ROI 폴리곤 반투명 채우기
    if roi_polygon is not None and len(roi_polygon) >= 3:
        overlay = base.copy()
        pts_arr = np.array(roi_polygon, dtype=np.int32)
        cv2.fillPoly(overlay, [pts_arr], ROI_FILL)
        cv2.addWeighted(overlay, 0.12, base, 0.88, 0, base)
        cv2.polylines(base, [pts_arr], True, ROI_EDGE, 1, cv2.LINE_AA)

    def _render():
        canvas = base.copy()
        n = len(lines)

        # 확정된 라인들 — 초록 단일 색상, 얇은 실선 유지
        for i, ((x1, y1), (x2, y2)) in enumerate(lines):
            cv2.line(canvas, (x1, y1), (x2, y2), LINE_COLOR_BGR, LINE_THICK, cv2.LINE_AA)
            # 끝점 마커
            cv2.circle(canvas, (x1, y1), ENDPOINT_R, LINE_COLOR_BGR, -1, cv2.LINE_AA)
            cv2.circle(canvas, (x2, y2), ENDPOINT_R, LINE_COLOR_BGR, -1, cv2.LINE_AA)
            cv2.circle(canvas, (x1, y1), ENDPOINT_R + 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(canvas, (x2, y2), ENDPOINT_R + 1, (255, 255, 255), 1, cv2.LINE_AA)
            # id 라벨 (중점) — 드로잉 중에는 id 가 필요하므로 숫자 표시 유지
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            label = str(i)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          M["label_font_sc"], M["label_font_th"])
            cv2.circle(canvas, (mx, my), max(tw, th) // 2 + M["label_pad"],
                       LINE_COLOR_BGR, -1, cv2.LINE_AA)
            cv2.putText(canvas, label, (mx - tw // 2, my + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, M["label_font_sc"],
                        (255, 255, 255), M["label_font_th"], cv2.LINE_AA)

        # 대기 중인 시작점 + 마우스 추적 미리보기
        if pending_start[0] is not None and not confirmed[0]:
            sx, sy = pending_start[0]
            cv2.line(canvas, (sx, sy), tuple(mouse_pos), LINE_COLOR_BGR, 1, cv2.LINE_AA)
            cv2.circle(canvas, (sx, sy), ENDPOINT_R, LINE_COLOR_BGR, -1, cv2.LINE_AA)
            cv2.circle(canvas, (sx, sy), ENDPOINT_R + 1, (255, 255, 255), 1, cv2.LINE_AA)

        # 상태바
        bar_h = M["bar_h"]
        canvas[:bar_h, :] = cv2.addWeighted(canvas[:bar_h, :], 0.4, np.zeros_like(canvas[:bar_h, :]), 0.6, 0)
        status = f"[{COUNTING_MODE}] Lines: {n}"
        if pending_start[0] is not None:
            status += " (click to finish current line)"
        if n < min_lines_required:
            status += f"  [need >={min_lines_required}]"
        elif rec_warn_msg and n < rec_lines_required:
            status += f"  [>= {rec_lines_required} recommended]"
        if has_prev[0]:
            status += "  [Loaded prev - C: clear]"
        cv2.putText(canvas, status, (10, M["bar_text_y"]),
                    cv2.FONT_HERSHEY_SIMPLEX, M["bar_font_sc"], (255, 255, 255),
                    M["bar_font_th"], cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, canvas)

    def _on_mouse(event, x, y, flags, param):
        mouse_pos[0], mouse_pos[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            if pending_start[0] is None:
                pending_start[0] = (x, y)
            else:
                lines.append((pending_start[0], (x, y)))
                pending_start[0] = None
            _render()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if pending_start[0] is not None:
                pending_start[0] = None
            elif lines:
                lines.pop()
            _render()
        elif event == cv2.EVENT_MOUSEMOVE:
            _render()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, frame.shape[1], frame.shape[0])
    cv2.setMouseCallback(WINDOW_NAME, _on_mouse)
    _render()

    ack_low_count = [False]  # 권장 미만 경고를 한 번 확인하면 재확인 없이 확정

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            skipped[0] = True
            break
        elif key in (13, 32):  # Enter / Space
            n = len(lines)
            if n < min_lines_required:
                print(f"  [lines] Need at least {min_lines_required} line(s) (current: {n})")
                continue
            if rec_warn_msg and n < rec_lines_required and not ack_low_count[0]:
                print(f"  [lines] Warning: {n} lines. {rec_warn_msg}. Press Enter again to confirm.")
                ack_low_count[0] = True
                continue
            confirmed[0] = True
            _render()
            cv2.waitKey(400)
            break
        elif key in (ord('r'), ord('R')):
            lines.clear()
            pending_start[0] = None
            has_prev[0] = False
            ack_low_count[0] = False
            _render()
        elif key in (ord('c'), ord('C')):
            lines.clear()
            pending_start[0] = None
            has_prev[0] = False
            ack_low_count[0] = False
            _render()

    cv2.destroyWindow(WINDOW_NAME)

    if skipped[0] or len(lines) < min_lines_required:
        return None

    return [
        {"id": i, "num_lines": 1, "coords": [[x1, y1, x2, y2]]}
        for i, ((x1, y1), (x2, y2)) in enumerate(lines)
    ]


# ═══════════════════════════════════════════════════════════════
# Per-video 통합 config (ROI + 카운팅 라인)
# ═══════════════════════════════════════════════════════════════

CONFIG_SCHEMA_VERSION = 1


def save_video_config(path, video_stem, src_wh, polylines_lst, roi_polygon):
    """per-video 통합 config를 JSON으로 저장.

    스키마:
    {
      "schema_version": 1,
      "video_stem": "sample",
      "saved_at": "2026-04-21T14:30:00",
      "canvas_size": "1920x1080",
      "polylines_lst": [...],
      "roi_polygon": [[x, y], ...] | null
    }
    """
    w, h = src_wh
    data = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "video_stem": video_stem,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "canvas_size": f"{int(w)}x{int(h)}",
        "polylines_lst": list(polylines_lst) if polylines_lst else [],
        "roi_polygon": (np.asarray(roi_polygon).tolist() if roi_polygon is not None else None),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  [config] 저장: {path}")


def load_video_config(path):
    """per-video 통합 config를 로드. 없으면 None."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [config] 로드 실패 {path}: {e}")
        return None
    return data


# ═══════════════════════════════════════════════════════════════
# 유틸리티 함수
# ═══════════════════════════════════════════════════════════════

def normalize_class(cls_name):
    """세분 차종명을 대분류 (car/bus/truck/bike/person) 로 정규화.

    - "truck_s", "truck_m", "truck_x" (또는 하이픈 변형) → "truck"
    - "bus_s", "bus_m" (또는 하이픈 변형)                → "bus"
    - 기타는 그대로 반환 ("car", "bike", "person" 등)
    """
    if cls_name.startswith("truck"):
        return "truck"
    if cls_name.startswith("bus"):
        return "bus"
    return cls_name


def is_hidden_class(cls_name):
    """렌더링에서 완전히 제외할 클래스인지 판정.
    원본 클래스명(bike, truck_s 등) 또는 정규화된 대분류 모두에서 체크.
    """
    if cls_name in HIDDEN_CLASSES:
        return True
    return normalize_class(cls_name) in HIDDEN_CLASSES


def get_display_name(cls_name):
    """차종 내부명을 그룹핑 설정에 따라 표시명으로 변환"""
    label_map = CLASS_EN if LABEL_LANG == "en" else CLASS_KOR
    if GROUP_TRUCK and cls_name.startswith("truck"):
        return "Truck" if LABEL_LANG == "en" else "화물차"
    if GROUP_BUS and cls_name.startswith("bus"):
        return "Bus" if LABEL_LANG == "en" else "버스"
    return label_map.get(cls_name, cls_name)


def group_class_counts(class_counts):
    """차종별 카운트를 그룹핑 설정에 따라 통합하여 [(display_name, count, color)] 반환"""
    grouped = {}
    order = []
    for cls_name in CLASS_DISPLAY_ORDER:
        count = class_counts.get(cls_name, 0)
        if count == 0:
            continue
        display = get_display_name(cls_name)
        if display in grouped:
            grouped[display] = (grouped[display][0] + count, grouped[display][1])
        else:
            color = CLASS_COLOR_MAP.get(cls_name, (200, 200, 200))
            grouped[display] = (count, color)
            order.append(display)
    # Stage-2 등 CLASS_DISPLAY_ORDER 에 없는 세분 클래스
    for cls_name in sorted(k for k in class_counts if class_counts[k] > 0):
        if cls_name in CLASS_DISPLAY_ORDER:
            continue
        display = get_display_name(cls_name)
        count = class_counts[cls_name]
        if display in grouped:
            grouped[display] = (grouped[display][0] + count, grouped[display][1])
        else:
            color = CLASS_COLOR_MAP.get(cls_name, (200, 200, 200))
            grouped[display] = (count, color)
            order.append(display)
    return [(name, grouped[name][0], grouped[name][1]) for name in order]


def _sanitize_filename_component(s, fallback="model"):
    """파일명에 안전한 토큰만 남기는 sanitize. 허용: A-Z a-z 0-9 . _ -
    공백/슬래시 등은 '-' 로 변환, 연속 '-' 축약, 앞뒤 '-' 제거.
    결과가 비면 fallback 반환.
    """
    if not s:
        return fallback
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(s))
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-._")
    return cleaned or fallback


def scale_lines(polylines_lst, src_wh, dst_wh):
    """polylines_lst를 src_wh 기준 좌표에서 dst_wh 기준 좌표로 스케일링.

    입력 포맷 (configs/{stem}.json 의 polylines_lst 필드와 동일):
        [{"id": 0, "num_lines": 1, "coords": [[x1, y1, x2, y2], ...]}, ...]

    반환:
        [(line_id: int, ((x1, y1, x2, y2), ...)), ...]
    """
    src_w, src_h = src_wh
    dst_w, dst_h = dst_wh
    scale_x = dst_w / src_w if src_w else 1.0
    scale_y = dst_h / src_h if src_h else 1.0

    counting_lines = []
    for item in polylines_lst:
        coords_lst = []
        for x1, y1, x2, y2 in item["coords"]:
            coords_lst.append((
                int(x1 * scale_x), int(y1 * scale_y),
                int(x2 * scale_x), int(y2 * scale_y)
            ))
        counting_lines.append((int(item["id"]), tuple(coords_lst)))
    return counting_lines


def class_height_weight(cls):
    """차종별 카운팅 기준점 y좌표 가중치 (utils__/utils.py 동일 로직)"""
    weights = {
        "truck_x": 0.7 + 0.2 * 0.606312,
        "truck_m": 0.7 + 0.2 * 0.607597,
        "truck_s": 0.7 + 0.2 * 0.235119,
        "bus_m":   0.7 + 0.2 * 0.6,
        "bus_s":   0.7 + 0.2 * 0.631794,
    }
    return weights.get(cls, 0.7)


def get_center_point(bbox, cls):
    """bbox와 차종에 따른 카운팅 기준점 좌표 계산 (utils__/utils.py 동일 로직 + 높이 오프셋)"""
    w = min(class_height_weight(cls) + CENTER_POINT_HEIGHT_OFFSET, 1.0)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    cx = int(bbox[0] + width / 2)
    cy = int(bbox[1] + height * w)
    return (cx, cy)


def filter_boxes_by_roi(boxes, label_mapping, roi_polygon):
    """ROI_OUTSIDE_BEHAVIOR == 'hide' 일 때 ROI 외부 박스를 통째로 제거.

    카운터/렌더러 모두에 동일한 필터를 일찍 적용하여, 외부에서 카운팅 라인을
    통과하는 객체가 카운트에 포함되지 않게 함과 동시에 시각화도 깔끔히 한다.

    Parameters
    ----------
    boxes : np.ndarray shape (N, 6) — [x1, y1, x2, y2, track_id, cls_idx]
    label_mapping : {int: str}
    roi_polygon   : np.ndarray shape (M, 2) | None

    Returns
    -------
    np.ndarray — 필터링된 boxes (외부거나 ROI 비활성이면 그대로).
    """
    if (not ROI_ENABLED
            or roi_polygon is None
            or ROI_OUTSIDE_BEHAVIOR != "hide"
            or len(boxes) == 0):
        return boxes

    keep = np.zeros(len(boxes), dtype=bool)
    for i, box in enumerate(boxes):
        cls_name = label_mapping.get(int(box[5]), "")
        cx, cy = get_center_point(box[:4], cls_name)
        keep[i] = cv2.pointPolygonTest(roi_polygon, (float(cx), float(cy)), False) >= 0
    return boxes[keep]


def _ccw(ax, ay, bx, by, cx, cy):
    return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)


def is_segments_intersecting(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    """두 선분의 교차 여부 판별 (utils__/intersect.py 동일 로직)"""
    if max(ax1, ax2) < min(bx1, bx2) or max(bx1, bx2) < min(ax1, ax2):
        return False
    if max(ay1, ay2) < min(by1, by2) or max(by1, by2) < min(ay1, ay2):
        return False
    return (_ccw(ax1, ay1, ax2, ay2, bx1, by1) != _ccw(ax1, ay1, ax2, ay2, bx2, by2)) and \
           (_ccw(bx1, by1, bx2, by2, ax1, ay1) != _ccw(bx1, by1, bx2, by2, ax2, ay2))


def check_hit(trajectory, counting_line_coords):
    """궤적선이 카운팅 라인과 교차하는지 검사 (utils__/intersect.py 동일 로직)"""
    x1, y1, x2, y2 = trajectory
    for ox1, oy1, ox2, oy2 in counting_line_coords:
        if is_segments_intersecting(x1, y1, x2, y2, ox1, oy1, ox2, oy2):
            return True, (ox1, oy1, ox2, oy2)
    return False, (0, 0, 0, 0)


def segment_intersection_point(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    """두 선분의 교차점 좌표를 반환"""
    dx1, dy1 = ax2 - ax1, ay2 - ay1
    dx2, dy2 = bx2 - bx1, by2 - by1
    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < 1e-10:
        return (ax1 + ax2) / 2, (ay1 + ay2) / 2
    t = ((bx1 - ax1) * dy2 - (by1 - ay1) * dx2) / denom
    return ax1 + t * dx1, ay1 + t * dy1


def cal_dist_sq_point_line(cx, cy, x1, y1, x2, y2):
    """점과 직선 사이의 거리 제곱 (utils__/intersect.py 동일 로직)"""
    dx, dy = x2 - x1, y2 - y1
    px, py = x1 - cx, y1 - cy
    cross_sq = (dx * py - dy * px) ** 2
    line_len_sq = dx * dx + dy * dy
    return cross_sq / line_len_sq if line_len_sq > 0 else 0


def cal_dist_sq_points(x1, y1, x2, y2):
    """두 점 사이의 거리 제곱"""
    return (x2 - x1) ** 2 + (y2 - y1) ** 2


# ═══════════════════════════════════════════════════════════════
# TrafficCounter (models__/object_counter.py 의 count_ver2 기반)
# ═══════════════════════════════════════════════════════════════

class TrackedObject:
    """개별 추적 객체의 카운팅 상태.

    COUNTING_MODE == "two_line"   : 두 라인 교차 조합으로 direction 집계 (CountedObject 동일 로직)
    COUNTING_MODE == "single_line": 각 라인 당 최초 교차 시 1회 카운트
    """
    def __init__(self, idx):
        self.idx = idx
        self.hit_lines = []
        self.directions = set()         # two_line 모드 전용
        self.counted_lines = set()      # single_line 모드 전용

    def process_hits_two_line(self, new_hits, cls_name, lane_class_counts, line_unique_tracks):
        self.hit_lines.extend(new_hits)
        if len(self.hit_lines) > 20:
            self.hit_lines = self.hit_lines[-20:]

        # 라인별 고유 track 집합 (HUD / 라인 라벨용) — 모든 hit 반영
        for h in new_hits:
            line_unique_tracks[h[1]].add(self.idx)

        for in_line, out_line in combinations(self.hit_lines, 2):
            in_idx, out_idx = in_line[1], out_line[1]
            if in_idx == out_idx:
                continue
            direction = f"{in_idx}->{out_idx}"
            if direction in self.directions:
                continue
            self.directions.add(direction)
            lane_class_counts[direction][cls_name] += 1

    def process_hits_single_line(self, new_hits, cls_name, line_class_counts,
                                 total_class_counts, line_unique_tracks):
        """single_line 모드:
          - line_class_counts[line_idx][cls] : 해당 라인을 처음 교차한 순간 +1 (라인별 고유)
          - total_class_counts[cls]          : 객체 생애 첫 라인 교차 때만 +1 (영상 전체 고유)
          - line_unique_tracks[line_idx]     : 해당 라인을 교차한 track id 집합
        """
        for h in new_hits:
            line_idx = h[1]
            line_unique_tracks[line_idx].add(self.idx)
            if line_idx in self.counted_lines:
                continue
            # 이 객체의 첫 라인 교차라면 전체 합에도 1회 기여
            if not self.counted_lines:
                total_class_counts[cls_name] += 1
            self.counted_lines.add(line_idx)
            line_class_counts[line_idx][cls_name] += 1


class SimpleCounter:
    """단일 스레드용 교통량 카운터.

    COUNTING_MODE 에 따라 집계 자료구조가 달라짐:
      - "two_line"   : lane_class_counts[direction][cls] = count
      - "single_line": line_class_counts[line_idx][cls] = count, total_class_counts[cls] = count
    line_unique_tracks[line_idx] = set(track_id)  (두 모드 공통, HUD/라인라벨용)
    """
    def __init__(self, counting_lines, label_mapping, target_fps=15):
        self.counting_lines = counting_lines
        self.label_mapping = label_mapping
        self._instances = {}
        self._untracked_frames = defaultdict(int)
        self.max_untracked_frames = 60 * target_fps
        # two_line 집계
        self.lane_class_counts = defaultdict(lambda: defaultdict(int))
        # single_line 집계
        self.line_class_counts = defaultdict(lambda: defaultdict(int))
        self.total_class_counts = defaultdict(int)
        # 라인별 고유 track (두 모드 공통)
        self.line_unique_tracks = defaultdict(set)
        self.hit_events = []  # 라인 교차 이벤트: [(line_idx, frame_idx, ix, iy), ...]
        self.previous_bboxes = None
        self.previous_time = 0

    def update(self, tracked_bboxes, video_time_ms, frame_idx=0):
        """
        tracked_bboxes: np.array [N, 6] - [x1, y1, x2, y2, track_id, cls]
        video_time_ms: 현재 프레임의 영상 시간 (ms)
        """
        for box in tracked_bboxes:
            self._untracked_frames[int(box[4])] = 0

        if self.previous_bboxes is not None and len(tracked_bboxes) > 0 and len(self.previous_bboxes) > 0:
            curr_map = {}
            for box in tracked_bboxes:
                tid = int(box[4])
                cls_name = self.label_mapping[int(box[5])]
                center = get_center_point(box[:4], cls_name)
                curr_map[tid] = (box, cls_name, center)

            prev_map = {}
            for box in self.previous_bboxes:
                tid = int(box[4])
                cls_name = self.label_mapping[int(box[5])]
                center = get_center_point(box[:4], cls_name)
                prev_map[tid] = (box, cls_name, center)

            matched_ids = set(curr_map.keys()) & set(prev_map.keys())

            for tid in matched_ids:
                _, cls_name, curr_center = curr_map[tid]
                _, _, prev_center = prev_map[tid]

                if tid not in self._instances:
                    self._instances[tid] = TrackedObject(tid)
                obj = self._instances[tid]

                trajectory = prev_center + curr_center

                hit_lines_info = []
                for line_idx, line_coords in self.counting_lines:
                    hit, hit_coord = check_hit(trajectory, line_coords)
                    if hit:
                        hit_lines_info.append((video_time_ms, line_idx, *hit_coord))
                        # 교차점 좌표 계산
                        ix, iy = segment_intersection_point(
                            *prev_center, *curr_center, *hit_coord)
                        self.hit_events.append((line_idx, frame_idx, ix, iy))

                if hit_lines_info:
                    hit_lines_info.sort(
                        reverse=True,
                        key=lambda h: cal_dist_sq_point_line(
                            curr_center[0], curr_center[1], h[2], h[3], h[4], h[5])
                    )

                    base_dist = cal_dist_sq_points(*prev_center, *curr_center)
                    time_diff = video_time_ms - self.previous_time

                    adjusted_hits = []
                    for h in hit_lines_info:
                        if base_dist > 0:
                            dist = cal_dist_sq_point_line(
                                curr_center[0], curr_center[1], h[2], h[3], h[4], h[5])
                            adjusted_time = h[0] + int(time_diff * (dist / base_dist))
                        else:
                            adjusted_time = h[0]
                        adjusted_hits.append((adjusted_time, h[1], *h[2:]))

                    if COUNTING_MODE == "single_line":
                        obj.process_hits_single_line(
                            adjusted_hits, cls_name,
                            self.line_class_counts,
                            self.total_class_counts,
                            self.line_unique_tracks,
                        )
                    else:
                        obj.process_hits_two_line(
                            adjusted_hits, cls_name,
                            self.lane_class_counts,
                            self.line_unique_tracks,
                        )

        # 오래된 객체 정리 (TrafficCounter.cleanup_untracked_objects 동일)
        del_list = []
        for tid in self._untracked_frames:
            if self._untracked_frames[tid] <= self.max_untracked_frames:
                self._untracked_frames[tid] += 1
            else:
                del_list.append(tid)
        for tid in del_list:
            if tid in self._instances:
                del self._instances[tid]
            del self._untracked_frames[tid]

        self.previous_bboxes = tracked_bboxes.copy() if len(tracked_bboxes) > 0 else tracked_bboxes
        self.previous_time = video_time_ms


# ═══════════════════════════════════════════════════════════════
# Skia Renderer (visualize.py SkiaRenderer 기반)
# ═══════════════════════════════════════════════════════════════

class SkiaCountingRenderer:
    def __init__(self, w, h, counting_lines, roi_polygon=None):
        self.w, self.h = w, h
        self.counting_lines = counting_lines

        # ── 해상도 자동 스케일 ───────────────────────────────────
        # 모든 픽셀 단위 시각화 요소에 곱해질 단일 배율.
        # 1080p 기준상수값들을 720p/4K 등에서도 시각적 비율 동일하게 유지.
        self.s = _viz_scale_factor(w, h)

        self.info = skia.ImageInfo.Make(w, h, skia.kRGBA_8888_ColorType, skia.kUnpremul_AlphaType)
        self.surface = skia.Surface.MakeRaster(self.info)
        self.pixmap = skia.Pixmap()

        self.class_skia_colors = {}
        for cls_name, (r, g, b) in CLASS_COLOR_MAP.items():
            self.class_skia_colors[cls_name] = skia.Color(r, g, b)

        # class_simple 모드 색상 (car/bus/truck 3색만)
        self.class_simple_skia_colors = {
            k: skia.Color(r, g, b) for k, (r, g, b) in CLASS_SIMPLE_COLORS_RGB.items()
        }
        self._class_simple_default_rgb = (200, 200, 200)

        # unified 모드용 색상
        _ur, _ug, _ub = UNIFIED_BBOX_COLOR
        self.unified_skia_color = skia.Color(_ur, _ug, _ub)
        self.unified_rgb = UNIFIED_BBOX_COLOR
        _tr, _tg, _tb = UNIFIED_TRAJ_COLOR
        self.unified_traj_color = skia.Color(_tr, _tg, _tb)
        self.unified_traj_rgb = UNIFIED_TRAJ_COLOR

        # ROI: 다각형 (np.array shape (N,2)) 또는 None
        self.roi_polygon = roi_polygon

        # 폰트 (한글 지원) — 모두 self.s 로 스케일링
        _tf_bold   = skia.Typeface("Noto Sans CJK KR", skia.FontStyle.Bold())
        _tf_normal = skia.Typeface("Noto Sans CJK KR", skia.FontStyle.Normal())
        self.font_title  = skia.Font(_tf_bold,   max(8.0, 16 * self.s))
        self.font_lane   = skia.Font(_tf_bold,   max(8.0, 13 * self.s))
        self.font_count  = skia.Font(_tf_normal, max(8.0, 12 * self.s))
        self.font_label  = skia.Font(_tf_bold,   max(8.0, 12 * self.s))
        self.font_line   = skia.Font(_tf_bold,   max(9.0, 14 * self.s))

        # ── 페인트 오브젝트 (visualize.py 동일) ──
        self.glow_paint = skia.Paint(AntiAlias=True)
        self.glow_paint.setStyle(skia.Paint.kFill_Style)
        self.glow_paint.setMaskFilter(
            skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, max(4.0, 18.0 * self.s)))

        self.box_paint = skia.Paint(AntiAlias=True)
        self.box_paint.setStyle(skia.Paint.kStroke_Style)
        self.box_paint.setStrokeWidth(max(1.0, 2.0 * self.s))

        # corner 스타일용 얇은 박스 페인트
        self.box_paint_thin = skia.Paint(AntiAlias=True)
        self.box_paint_thin.setStyle(skia.Paint.kStroke_Style)
        self.box_paint_thin.setStrokeWidth(max(0.5, BBOX_STROKE_WIDTH * self.s))

        self.corner_paint = skia.Paint(AntiAlias=True)
        self.corner_paint.setStyle(skia.Paint.kStroke_Style)
        self.corner_paint.setStrokeWidth(max(1.0, BBOX_CORNER_WIDTH * self.s))
        self.corner_paint.setStrokeCap(skia.Paint.kButt_Cap)

        # 반투명 오버레이 채우기 페인트
        self.fill_paint = skia.Paint(AntiAlias=True)
        self.fill_paint.setStyle(skia.Paint.kFill_Style)

        self.dot_paint = skia.Paint(AntiAlias=True)

        self.trace_paint = skia.Paint(AntiAlias=True)
        self.trace_paint.setStyle(skia.Paint.kStroke_Style)
        self.trace_paint.setStrokeCap(skia.Paint.kRound_Cap)

        self.bg_paint = skia.Paint(AntiAlias=True)

        self.shadow_paint = skia.Paint(AntiAlias=True)
        self.shadow_paint.setColor(skia.Color(0, 0, 0))
        self.shadow_paint.setAlphaf(0.4)

        self.text_paint = skia.Paint(AntiAlias=True)
        self.text_paint.setColor(skia.Color(255, 255, 255))

        # ── 카운팅 라인 페인트 (StrokeWidth 는 _draw_counting_lines 에서 매 프레임 재설정) ──
        self.line_paint = skia.Paint(AntiAlias=True)
        self.line_paint.setStyle(skia.Paint.kStroke_Style)
        self.line_paint.setStrokeWidth(max(1.0, 2.5 * self.s))
        self.line_paint.setStrokeCap(skia.Paint.kRound_Cap)

        self.line_glow_paint = skia.Paint(AntiAlias=True)
        self.line_glow_paint.setStyle(skia.Paint.kStroke_Style)
        self.line_glow_paint.setStrokeWidth(max(2.0, 8.0 * self.s))
        self.line_glow_paint.setMaskFilter(
            skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, max(2.0, 6.0 * self.s)))

        # ── HUD 패널 페인트 ──
        self.hud_bg_paint = skia.Paint(AntiAlias=True)
        self.hud_bg_paint.setColor(skia.Color(15, 15, 25))
        self.hud_bg_paint.setAlphaf(0.78)

        self.hud_border_paint = skia.Paint(AntiAlias=True)
        self.hud_border_paint.setStyle(skia.Paint.kStroke_Style)
        self.hud_border_paint.setStrokeWidth(max(0.5, 1.0 * self.s))
        self.hud_border_paint.setColor(skia.Color(100, 120, 160))
        self.hud_border_paint.setAlphaf(0.4)

        # trajectory 히스토리
        self.trace_history = defaultdict(list)
        self.trace_max = 80

    def _cls_color(self, cls_name):
        if COLOR_MODE == "unified":
            return self.unified_skia_color
        if COLOR_MODE == "class_simple":
            key = normalize_class(cls_name)
            return self.class_simple_skia_colors.get(key, skia.Color(*self._class_simple_default_rgb))
        return self.class_skia_colors.get(cls_name, skia.Color(200, 200, 200))

    def _cls_rgb(self, cls_name):
        if COLOR_MODE == "unified":
            return self.unified_rgb
        if COLOR_MODE == "class_simple":
            key = normalize_class(cls_name)
            return CLASS_SIMPLE_COLORS_RGB.get(key, self._class_simple_default_rgb)
        return CLASS_COLOR_MAP.get(cls_name, (200, 200, 200))

    def _traj_color(self, cls_name):
        # trajectory 도 bbox 와 동일한 팔레트 사용 (class_simple 일 때 3색만)
        if COLOR_MODE == "unified":
            return self.unified_traj_color
        if COLOR_MODE == "class_simple":
            key = normalize_class(cls_name)
            return self.class_simple_skia_colors.get(key, skia.Color(*self._class_simple_default_rgb))
        return self.class_skia_colors.get(cls_name, skia.Color(200, 200, 200))

    def _traj_rgb(self, cls_name):
        if COLOR_MODE == "unified":
            return self.unified_traj_rgb
        if COLOR_MODE == "class_simple":
            key = normalize_class(cls_name)
            return CLASS_SIMPLE_COLORS_RGB.get(key, self._class_simple_default_rgb)
        return CLASS_COLOR_MAP.get(cls_name, (200, 200, 200))

    def _is_in_roi(self, cx, cy):
        """객체 중심점이 ROI 다각형 내부인지 판정"""
        if not ROI_ENABLED or self.roi_polygon is None:
            return True
        return cv2.pointPolygonTest(self.roi_polygon, (float(cx), float(cy)), False) >= 0

    def _roi_alpha(self, cx, cy):
        """ROI 내/외부에 따른 alpha 값 반환"""
        if self._is_in_roi(cx, cy):
            return ROI_INSIDE_ALPHA
        return ROI_OUTSIDE_ALPHA

    # ──────────────────────────────────────────────────────────
    # 메인 draw
    # ──────────────────────────────────────────────────────────
    def draw(self, frame, boxes, label_mapping, counter_state, hit_events=None, frame_idx=0,
             panel_title=None):
        """
        frame: BGR numpy array
        boxes: np.array [N, 6] - [x1, y1, x2, y2, track_id, cls]
        label_mapping: {int: str}
        counter_state: dict — SimpleCounter 상태 스냅샷. 필요한 키:
            - "lane_class_counts": {direction: {cls_name: count}}     (two_line)
            - "line_class_counts": {line_idx: {cls_name: count}}      (single_line)
            - "total_class_counts": {cls_name: count}                 (single_line)
            - "line_unique_tracks": {line_idx: set(tid)}              (공통, 라인 라벨용)
        hit_events: [(line_idx, event_frame_idx, ix, iy), ...] — SHOW_LINE_HIT_EFFECT / SHOW_CROSS_MARKER 용
        frame_idx  : 현재 프레임 번호
        panel_title: 비교 모드에서 각 패널 상단에 표시할 모델명 (None 이면 미표시)
        """
        lane_class_counts   = counter_state.get("lane_class_counts", {})
        line_class_counts   = counter_state.get("line_class_counts", {})
        total_class_counts  = counter_state.get("total_class_counts", {})
        line_unique_tracks  = counter_state.get("line_unique_tracks", {})
        canvas = self.surface.getCanvas()

        rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        canvas.writePixels(self.info, rgba.tobytes(), self.w * 4, 0, 0)

        # 1) 카운팅 라인 (교차 이벤트 이펙트 포함)
        # 라인별 최대 이펙트 강도 및 교차점 수집
        effect_duration = 20
        hit_intensity = {}
        active_intersections = []  # [(ix, iy), ...]
        if (SHOW_LINE_HIT_EFFECT or SHOW_CROSS_MARKER) and hit_events:
            for event in hit_events:
                line_idx, event_frame, ix, iy = event
                age = frame_idx - event_frame
                if 0 <= age < effect_duration:
                    if SHOW_LINE_HIT_EFFECT:
                        intensity = 1.0 - (age / effect_duration)
                        hit_intensity[line_idx] = max(hit_intensity.get(line_idx, 0), intensity)
                    if SHOW_CROSS_MARKER:
                        active_intersections.append((ix, iy))
        self._draw_counting_lines(canvas, hit_intensity, line_unique_tracks)

        # 교차점 마커 (정적 표시 - 모든 교차점 유지)
        marker_r_inner = max(2.0, 5.0 * self.s)
        marker_r_outer = max(4.0, 10.0 * self.s)
        marker_ring_w  = max(0.8, 1.5 * self.s)
        for ix, iy in active_intersections:
            marker_paint = skia.Paint(AntiAlias=True)
            marker_paint.setColor(skia.Color(255, 255, 255))
            marker_paint.setAlphaf(0.6)
            canvas.drawCircle(float(ix), float(iy), marker_r_inner, marker_paint)

            ring_paint = skia.Paint(AntiAlias=True)
            ring_paint.setStyle(skia.Paint.kStroke_Style)
            ring_paint.setStrokeWidth(marker_ring_w)
            ring_paint.setColor(skia.Color(255, 255, 255))
            ring_paint.setAlphaf(0.3)
            canvas.drawCircle(float(ix), float(iy), marker_r_outer, ring_paint)

        # 1.5) ROI 경계선
        if ROI_SHOW_BOUNDARY and ROI_ENABLED and self.roi_polygon is not None:
            pts = self.roi_polygon
            n = len(pts)
            # 반투명 경계선
            roi_line_paint = skia.Paint(AntiAlias=True)
            roi_line_paint.setStyle(skia.Paint.kStroke_Style)
            roi_line_paint.setColor(skia.Color(255, 255, 255))
            roi_line_paint.setAlphaf(ROI_BOUNDARY_ALPHA)
            roi_line_paint.setStrokeWidth(max(0.8, ROI_BOUNDARY_WIDTH * self.s))
            roi_path = skia.Path()
            roi_path.moveTo(float(pts[0][0]), float(pts[0][1]))
            for i in range(1, n):
                roi_path.lineTo(float(pts[i][0]), float(pts[i][1]))
            roi_path.close()
            canvas.drawPath(roi_path, roi_line_paint)

            # ROI 외부를 살짝 어둡게 (반투명 마스크)
            roi_dim_paint = skia.Paint(AntiAlias=True)
            roi_dim_paint.setColor(skia.Color(0, 0, 0))
            roi_dim_paint.setAlphaf(ROI_DIM_ALPHA)
            outer_path = skia.Path()
            outer_path.addRect(skia.Rect(0, 0, self.w, self.h))
            outer_path.addPath(roi_path)
            outer_path.setFillType(skia.PathFillType.kEvenOdd)
            canvas.drawPath(outer_path, roi_dim_paint)

        # 2) 1패스: 데이터 수집 + trace_history 업데이트
        draw_data = []  # [(x1,y1,x2,y2, col, r,g,b, alpha, in_roi, bcx,bcy, tid, cls_name)]
        for box in boxes:
            x1, y1, x2, y2 = map(float, box[:4])
            tid = int(box[4])
            cls_idx = int(box[5])
            cls_name = label_mapping.get(cls_idx, f"cls_{cls_idx}")

            # bike / person 등 숨김 클래스: bbox, trajectory, label 모두 스킵
            if is_hidden_class(cls_name):
                continue

            col = self._cls_color(cls_name)
            r, g, b = self._cls_rgb(cls_name)

            obj_cx, obj_cy = get_center_point(box[:4], cls_name)
            alpha = self._roi_alpha(obj_cx, obj_cy)
            in_roi = self._is_in_roi(obj_cx, obj_cy)

            bcx, bcy = get_center_point(box[:4], cls_name)

            # trace_history 업데이트 (궤적 드로잉 전에 수행)
            if SHOW_TRAJECTORY and tid >= 0:
                self.trace_history[tid].append((bcx, bcy))
                if len(self.trace_history[tid]) > self.trace_max:
                    self.trace_history[tid] = self.trace_history[tid][-self.trace_max:]

            draw_data.append((x1, y1, x2, y2, col, r, g, b, alpha, in_roi, bcx, bcy, tid, cls_name))

        # 2a) bbox + 중심점
        # 해상도 스케일 적용한 코너/글로우 픽셀 메트릭 (프레임 루프 밖에서 1회 산출)
        s = self.s
        corner_max_w = max(1.0, BBOX_CORNER_WIDTH * s)
        corner_min_w = max(0.5, BBOX_CORNER_MIN_WIDTH * s)
        corner_max_l = max(4.0, BBOX_CORNER_LENGTH * s)
        corner_min_l = max(3.0, BBOX_CORNER_MIN_LEN * s)
        corner_ref   = max(1.0, BBOX_CORNER_REF_SIZE * s)
        glow_pad     = max(2.0, 6.0 * s)
        glow_round   = max(4.0, 12.0 * s)
        rrect_round  = max(3.0, 8.0  * s)

        for x1, y1, x2, y2, col, r, g, b, alpha, in_roi, bcx, bcy, tid, cls_name in draw_data:
            bw, bh = x2 - x1, y2 - y1

            if BBOX_STYLE == "corner":
                # ── corner 스타일: 얇은 직각 박스 + 코너 ──
                self.box_paint_thin.setColor(col)
                self.box_paint_thin.setAlphaf(alpha)
                canvas.drawRect(skia.Rect(x1, y1, x2, y2), self.box_paint_thin)

                # 코너 강조
                if BBOX_CORNER_DYNAMIC:
                    scale = min(1.0, min(bw, bh) / corner_ref)
                    cw = corner_min_w + (corner_max_w - corner_min_w) * scale
                    clen = max(corner_min_l, min(bw * BBOX_CORNER_RATIO, bh * BBOX_CORNER_RATIO, corner_max_l))
                else:
                    cw = corner_max_w
                    clen = min(bw * BBOX_CORNER_RATIO, bh * BBOX_CORNER_RATIO, corner_max_l)
                self.corner_paint.setColor(col)
                self.corner_paint.setStrokeWidth(cw)
                self.corner_paint.setAlphaf(alpha)
                for cx, cy, dx, dy in [
                    (x1, y1, 1, 1), (x2, y1, -1, 1),
                    (x1, y2, 1, -1), (x2, y2, -1, -1),
                ]:
                    p = skia.Path()
                    p.moveTo(cx, cy + dy * clen)
                    p.lineTo(cx, cy)
                    p.lineTo(cx + dx * clen, cy)
                    canvas.drawPath(p, self.corner_paint)
            else:
                # ── default 스타일: 글로우 + 둥근 모서리 + 코너 ──
                self.glow_paint.setColor(col)
                self.glow_paint.setAlphaf(0.12 * alpha)
                canvas.drawRRect(
                    skia.RRect.MakeRectXY(
                        skia.Rect(x1 - glow_pad, y1 - glow_pad, x2 + glow_pad, y2 + glow_pad),
                        glow_round, glow_round),
                    self.glow_paint,
                )

                self.box_paint.setColor(col)
                self.box_paint.setAlphaf(alpha)
                canvas.drawRRect(
                    skia.RRect.MakeRectXY(skia.Rect(x1, y1, x2, y2), rrect_round, rrect_round),
                    self.box_paint,
                )

                # 코너 인디케이터
                if BBOX_CORNER_DYNAMIC:
                    scale = min(1.0, min(bw, bh) / corner_ref)
                    cw = corner_min_w + (corner_max_w - corner_min_w) * scale
                    clen = max(corner_min_l, min(bw * BBOX_CORNER_RATIO, bh * BBOX_CORNER_RATIO, corner_max_l))
                else:
                    cw = corner_max_w
                    clen = min(bw * BBOX_CORNER_RATIO, bh * BBOX_CORNER_RATIO, corner_max_l)
                self.corner_paint.setColor(col)
                self.corner_paint.setStrokeWidth(cw)
                self.corner_paint.setAlphaf(alpha)
                for cx, cy, dx, dy in [
                    (x1, y1, 1, 1), (x2, y1, -1, 1),
                    (x1, y2, 1, -1), (x2, y2, -1, -1),
                ]:
                    p = skia.Path()
                    p.moveTo(cx, cy + dy * clen)
                    p.lineTo(cx, cy)
                    p.lineTo(cx + dx * clen, cy)
                    canvas.drawPath(p, self.corner_paint)

            # 반투명 오버레이 채우기
            if BBOX_FILL_OVERLAY:
                self.fill_paint.setColor(col)
                self.fill_paint.setAlphaf(BBOX_FILL_ALPHA * alpha)
                canvas.drawRect(skia.Rect(x1, y1, x2, y2), self.fill_paint)

            # 중심점
            if SHOW_CENTER_POINT:
                self.dot_paint.setColor(col)
                self.dot_paint.setAlphaf(alpha)
                canvas.drawCircle(bcx, bcy, max(1.0, CENTER_POINT_RADIUS * s), self.dot_paint)

        # 2b) 궤적 — 모든 bbox 위에 그림
        traj_max_w = max(0.6, TRAJECTORY_WIDTH * s)
        traj_min_w = max(0.3, TRAJECTORY_TAPER_MIN_WIDTH * s)
        for x1, y1, x2, y2, col, r, g, b, alpha, in_roi, bcx, bcy, tid, cls_name in draw_data:
            if not (SHOW_TRAJECTORY and tid >= 0):
                continue
            pts = self.trace_history.get(tid, [])
            if len(pts) < 2:
                continue
            traj_col = self._traj_color(cls_name)
            n = len(pts)

            # 세그먼트별 alpha/width 계산 후 glow → core 순으로 그림
            for i in range(n - 1):
                t = (i + 1) / n  # 0(오래됨) → 1(최근)

                seg_alpha = TRAJECTORY_ALPHA * (t if TRAJECTORY_FADE else 1.0) * alpha
                core_w = (traj_min_w + (traj_max_w - traj_min_w) * t
                          if TRAJECTORY_TAPER else traj_max_w)

                # 글로우 레이어
                if TRAJECTORY_GLOW:
                    glow_paint = skia.Paint(AntiAlias=True)
                    glow_paint.setStyle(skia.Paint.kStroke_Style)
                    glow_paint.setStrokeCap(skia.Paint.kRound_Cap)
                    glow_paint.setColor(traj_col)
                    glow_paint.setStrokeWidth(core_w * TRAJECTORY_GLOW_WIDTH_FACTOR)
                    glow_paint.setAlphaf(TRAJECTORY_GLOW_ALPHA * (t if TRAJECTORY_FADE else 1.0) * alpha)
                    canvas.drawLine(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1], glow_paint)

                # 코어 레이어
                self.trace_paint.setColor(traj_col)
                self.trace_paint.setAlphaf(seg_alpha)
                self.trace_paint.setStrokeWidth(core_w)
                canvas.drawLine(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1], self.trace_paint)

        # 2c) 라벨 — 궤적 위에 그림
        lh = max(12.0, 20.0 * s)
        pill_pad_x = max(6.0, 14.0 * s)
        pill_round = max(3.0, 6.0 * s)
        text_off_x = max(3.0, 7.0 * s)
        text_off_y = max(8.0, 14.0 * s)
        for x1, y1, x2, y2, col, r, g, b, alpha, in_roi, bcx, bcy, tid, cls_name in draw_data:
            if not (in_roi or ROI_SHOW_LABEL):
                continue
            label = get_display_name(cls_name)
            if SHOW_TRACK_ID and tid >= 0:
                label = f"#{tid} {label}"
            text_w = self.font_label.measureText(label)
            lx, ly = x1, y1 - lh - 2
            if ly < 0:
                ly = y2 + 2

            pill = skia.RRect.MakeRectXY(
                skia.Rect(lx, ly, lx + text_w + pill_pad_x, ly + lh), pill_round, pill_round)
            self.bg_paint.setShader(skia.GradientShader.MakeLinear(
                points=[(lx, ly), (lx + text_w + pill_pad_x, ly)],
                colors=[col, skia.Color(min(255, r + 40), min(255, g + 40), min(255, b + 40))],
            ))
            self.bg_paint.setAlphaf(0.88 * alpha)
            canvas.drawRRect(pill, self.bg_paint)

            self.shadow_paint.setAlphaf(0.4 * alpha)
            canvas.drawString(label, lx + text_off_x + 1, ly + text_off_y + 1,
                              self.font_label, self.shadow_paint)
            self.text_paint.setAlphaf(alpha)
            canvas.drawString(label, lx + text_off_x, ly + text_off_y,
                              self.font_label, self.text_paint)

        # 3) HUD 패널 (HUD_MODE 분기는 _draw_hud 내부)
        if HUD_MODE != "off":
            self._draw_hud(canvas, counter_state, panel_title=panel_title)

        # 캔버스 → BGR numpy
        self.surface.peekPixels(self.pixmap)
        arr = np.frombuffer(self.pixmap.tobytes(), dtype=np.uint8).reshape(self.h, self.w, 4)
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

    # ──────────────────────────────────────────────────────────
    # 카운팅 라인 시각화 (초록 단일색, 두꺼운 반투명 실선 + 라인별 통과 수 표시)
    # ──────────────────────────────────────────────────────────
    def _draw_counting_lines(self, canvas, hit_intensity=None, line_unique_tracks=None):
        lr, lg, lb = LINE_COLOR_RGB
        line_color = skia.Color(lr, lg, lb)
        if hit_intensity is None:
            hit_intensity = {}
        if line_unique_tracks is None:
            line_unique_tracks = {}

        # 해상도 자동 스케일 적용한 굵기/반지름 (1080p 기준값 × self.s)
        s = self.s
        line_thick = max(2.0, LINE_VIZ_THICKNESS * s)
        glow_thick = max(4.0, LINE_VIZ_THICKNESS * 2.0 * s)
        circle_rad_min = max(8.0, LINE_COUNT_CIRCLE_RAD * s)
        label_pad      = max(4.0, 8.0 * s)
        border_w       = max(0.8, 1.2 * s)
        text_y_off     = max(3.0, 5.0 * s)
        flash_blur_base = max(2.0, 4.0 * s)
        flash_blur_dyn  = max(4.0, 12.0 * s)

        self.line_paint.setStrokeWidth(line_thick)
        self.line_glow_paint.setStrokeWidth(glow_thick)

        for line_idx, line_coords in self.counting_lines:
            intensity = hit_intensity.get(line_idx, 0)
            pass_count = len(line_unique_tracks.get(line_idx, ()))

            for x1, y1, x2, y2 in line_coords:
                if intensity > 0:
                    # 히트 이펙트 (SHOW_LINE_HIT_EFFECT=True 일 때만 hit_intensity 가 채워짐)
                    flash_paint = skia.Paint(AntiAlias=True)
                    flash_paint.setStyle(skia.Paint.kStroke_Style)
                    flash_paint.setStrokeWidth(line_thick * (1.5 + intensity))
                    flash_paint.setColor(skia.Color(
                        min(255, lr + int((255 - lr) * intensity)),
                        min(255, lg + int((255 - lg) * intensity)),
                        min(255, lb + int((255 - lb) * intensity)),
                    ))
                    flash_paint.setAlphaf(0.5 * intensity)
                    flash_paint.setMaskFilter(
                        skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle,
                                                  flash_blur_dyn * intensity + flash_blur_base))
                    canvas.drawLine(x1, y1, x2, y2, flash_paint)

                # 기본 글로우 (반투명, 굵은 blur)
                self.line_glow_paint.setColor(line_color)
                self.line_glow_paint.setAlphaf(LINE_VIZ_GLOW_ALPHA)
                canvas.drawLine(x1, y1, x2, y2, self.line_glow_paint)

                # 기본 실선 (굵고 반투명)
                self.line_paint.setColor(line_color)
                self.line_paint.setAlphaf(LINE_VIZ_ALPHA)
                canvas.drawLine(x1, y1, x2, y2, self.line_paint)

                # 라인 중점에 "통과 차량 수" 라벨 (line_idx 숫자는 표시 안 함)
                # 배경 원은 LINE_COUNT_LABEL_ALPHA 로 조절 가능 (기본 매우 투명).
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                num_str = str(pass_count)
                num_w = self.font_line.measureText(num_str)
                radius = max(circle_rad_min, num_w / 2 + label_pad)

                # 원형 배경 (초록, 반투명) — 기본은 도로가 잘 보이도록 매우 투명
                circle_paint = skia.Paint(AntiAlias=True)
                circle_paint.setColor(line_color)
                circle_paint.setAlphaf(LINE_COUNT_LABEL_ALPHA)
                canvas.drawCircle(mid_x, mid_y, radius, circle_paint)

                # 원형 테두리 (흰색 얇은) — 가독성 보조
                circle_border = skia.Paint(AntiAlias=True)
                circle_border.setStyle(skia.Paint.kStroke_Style)
                circle_border.setStrokeWidth(border_w)
                circle_border.setColor(skia.Color(255, 255, 255))
                circle_border.setAlphaf(LINE_COUNT_LABEL_BORDER_ALPHA)
                canvas.drawCircle(mid_x, mid_y, radius, circle_border)

                # 숫자 (그림자 1px 살짝 더해 투명 배경에서도 가독성 확보)
                shadow = skia.Paint(AntiAlias=True)
                shadow.setColor(skia.Color(0, 0, 0))
                shadow.setAlphaf(0.55)
                canvas.drawString(num_str, mid_x - num_w / 2 + 1, mid_y + text_y_off + 1,
                                  self.font_line, shadow)
                num_paint = skia.Paint(AntiAlias=True)
                num_paint.setColor(skia.Color(255, 255, 255))
                canvas.drawString(num_str, mid_x - num_w / 2, mid_y + text_y_off,
                                  self.font_line, num_paint)

    # ──────────────────────────────────────────────────────────
    # HUD 패널 (좌상단)
    # ──────────────────────────────────────────────────────────
    def _aggregate_class_totals(self, counter_state):
        """COUNTING_MODE 에 맞춰 라인/차선별 카운트를 차종 키로 합산.

        - single_line : line_class_counts 의 모든 line_idx 합산
                        (한 차량이 N개 라인 통과 시 N회 누적)
        - two_line    : lane_class_counts 의 모든 direction 합산
        """
        aggregated = defaultdict(int)
        if COUNTING_MODE == "single_line":
            for _line_idx, cc in counter_state.get("line_class_counts", {}).items():
                for cls, n in cc.items():
                    aggregated[cls] += n
        else:
            for _dir, cc in counter_state.get("lane_class_counts", {}).items():
                for cls, n in cc.items():
                    aggregated[cls] += n
        return aggregated

    def _draw_hud(self, canvas, counter_state, panel_title=None):
        """HUD_MODE 에 따라 totals / full 분기."""
        if HUD_MODE == "totals":
            self._draw_hud_totals(canvas, counter_state, panel_title=panel_title)
            return
        # HUD_MODE == "full" — 라인/차선 × 차종 상세 (기존 동작)
        self._draw_hud_full(canvas, counter_state, panel_title=panel_title)

    # ─────── HUD: totals 레이아웃 (차종별 합계만) ───────
    def _draw_hud_totals(self, canvas, counter_state, panel_title=None):
        """좌상단에 차종별 합계만 단순하게 표시.

        - 0 건 차종은 숨김
        - HIDDEN_CLASSES (bike/person 등) 자동 제외
        - 색상 인디케이터는 _cls_rgb (COLOR_MODE 적용 — class_simple 이면 3색)
        - 라벨은 get_display_name (모델 출력 그대로의 세분류)
        - 마지막에 Grand Total 행
        """
        s = self.s
        aggregated = self._aggregate_class_totals(counter_state)

        # 정렬: CLASS_DISPLAY_ORDER 우선, 외 클래스(예: Stage-2 12종)는 알파벳 순
        ordered_keys = [c for c in CLASS_DISPLAY_ORDER if c in aggregated]
        extra_keys   = sorted(c for c in aggregated if c not in CLASS_DISPLAY_ORDER)
        rows = []
        for cls in ordered_keys + extra_keys:
            n = aggregated[cls]
            if n <= 0 or is_hidden_class(cls):
                continue
            rows.append((cls, n, get_display_name(cls), self._cls_rgb(cls)))
        grand_total = sum(n for _, n, _, _ in rows)

        # 레이아웃 메트릭
        margin = max(8.0,  16.0 * s)
        pad    = max(6.0,  14.0 * s)
        x0, y0 = margin, margin

        title_h    = max(16.0, 28.0 * s)
        subtitle_h = max(12.0, 20.0 * s) if panel_title else 0
        divider_h  = max(6.0,  10.0 * s)
        row_h      = max(12.0, 20.0 * s)
        section_gap = max(3.0, 6.0  * s)

        title_off = max(10.0, 16.0 * s)
        sub_off   = max(8.0,  13.0 * s)
        row_off   = max(8.0,  14.0 * s)
        ind_off_x = max(6.0,  10.0 * s)
        ind_r     = max(2.0,  4.0  * s)
        cls_off_x = max(12.0, 20.0 * s)
        sep_w     = max(0.5,  1.0  * s)

        n_rows = max(1, len(rows))
        total_h = (pad + title_h + subtitle_h + divider_h
                   + row_h * n_rows + section_gap + row_h + pad)
        panel_w = max(110.0, 168.0 * s)
        panel_round = max(5.0, 10.0 * s)

        # 패널 배경
        panel_rect = skia.RRect.MakeRectXY(
            skia.Rect(x0, y0, x0 + panel_w, y0 + total_h), panel_round, panel_round)
        canvas.drawRRect(panel_rect, self.hud_bg_paint)
        canvas.drawRRect(panel_rect, self.hud_border_paint)

        cx, cy = x0 + pad, y0 + pad
        right_edge = x0 + panel_w - pad

        # 타이틀
        title_paint = skia.Paint(AntiAlias=True)
        title_paint.setColor(skia.Color(230, 240, 255))
        canvas.drawString("Total Counts", cx, cy + title_off, self.font_title, title_paint)
        cy += title_h

        # 부제 (비교 모드 모델명)
        if panel_title:
            sub_paint = skia.Paint(AntiAlias=True)
            sub_paint.setColor(skia.Color(170, 200, 240))
            canvas.drawString(panel_title, cx, cy + sub_off, self.font_lane, sub_paint)
            cy += subtitle_h

        # 구분선
        sep_paint = skia.Paint(AntiAlias=True)
        sep_paint.setColor(skia.Color(100, 120, 160))
        sep_paint.setAlphaf(0.3)
        sep_paint.setStrokeWidth(sep_w)
        canvas.drawLine(cx, cy, right_edge, cy, sep_paint)
        cy += divider_h

        if rows:
            for _cls, n, disp, (r, g, b) in rows:
                ind_paint = skia.Paint(AntiAlias=True)
                ind_paint.setColor(skia.Color(r, g, b))
                canvas.drawCircle(cx + ind_off_x, cy + row_h / 2, ind_r, ind_paint)
                cls_paint = skia.Paint(AntiAlias=True)
                cls_paint.setColor(skia.Color(200, 210, 230))
                canvas.drawString(disp, cx + cls_off_x, cy + row_off, self.font_count, cls_paint)

                count_str = str(n)
                count_w = self.font_count.measureText(count_str)
                count_paint = skia.Paint(AntiAlias=True)
                count_paint.setColor(skia.Color(255, 255, 255))
                canvas.drawString(count_str, right_edge - count_w, cy + row_off,
                                  self.font_count, count_paint)
                cy += row_h
        else:
            empty_paint = skia.Paint(AntiAlias=True)
            empty_paint.setColor(skia.Color(100, 110, 130))
            canvas.drawString("  -", cx + ind_off_x, cy + row_off, self.font_count, empty_paint)
            cy += row_h

        cy += section_gap

        # Grand Total 행 (구분선 위)
        canvas.drawLine(cx, cy, right_edge, cy, sep_paint)
        cy += max(2.0, 4.0 * s)

        total_label_paint = skia.Paint(AntiAlias=True)
        total_label_paint.setColor(skia.Color(140, 180, 255))
        canvas.drawString("Total", cx, cy + row_off, self.font_lane, total_label_paint)

        gt_str = str(grand_total)
        gt_w = self.font_lane.measureText(gt_str)
        gt_paint = skia.Paint(AntiAlias=True)
        gt_paint.setColor(skia.Color(255, 255, 255))
        canvas.drawString(gt_str, right_edge - gt_w, cy + row_off, self.font_lane, gt_paint)

    # ─────── HUD: full 레이아웃 (라인/차선 × 차종 상세) ───────
    def _draw_hud_full(self, canvas, counter_state, panel_title=None):
        """
        counter_state 에서 COUNTING_MODE 에 맞는 필드만 꺼내 사용.
        panel_title 이 주어지면 HUD 타이틀 아래에 부제(모델명) 로 추가 표시.
        """
        s = self.s
        margin = max(8.0, 16.0 * s)
        pad    = max(6.0, 14.0 * s)
        x0, y0 = margin, margin

        title_h          = max(16.0, 28.0 * s)
        subtitle_h       = max(12.0, 20.0 * s) if panel_title else 0
        divider_h        = max(6.0,  10.0 * s)
        section_header_h = max(14.0, 24.0 * s)
        class_row_h      = max(12.0, 20.0 * s)
        section_gap      = max(3.0,  6.0  * s)

        # ── 섹션 데이터 구성 ──
        if COUNTING_MODE == "single_line":
            line_class_counts = counter_state.get("line_class_counts", {})
            total_class_counts = counter_state.get("total_class_counts", {})
            line_unique_tracks = counter_state.get("line_unique_tracks", {})

            sections = []  # [(section_name, total, grouped_rows [(disp, count, (r,g,b))])]
            # 라인별 섹션 (라인 idx 오름차순)
            for line_idx, _ in self.counting_lines:
                label = SINGLE_LINE_LABELS.get(line_idx, f"Line {line_idx}")
                class_counts = line_class_counts.get(line_idx, {})
                grouped = group_class_counts(class_counts)
                total = len(line_unique_tracks.get(line_idx, ()))  # 고유 track 수
                sections.append((label, total, grouped))

            # 전체 합산 섹션
            total_grouped = group_class_counts(total_class_counts)
            total_all = sum(c for _, c, _ in total_grouped)
            sections.append(("Total", total_all, total_grouped))
            hud_title = "Traffic Info (single-line)"

        else:
            lane_class_counts = counter_state.get("lane_class_counts", {})
            sections = []
            for direction, lane_name in LANE_MAP.items():
                class_counts = lane_class_counts.get(direction, {})
                grouped = group_class_counts(class_counts)
                total = sum(c for _, c, _ in grouped)
                sections.append((lane_name, total, grouped))
            hud_title = "Traffic Info"

        # 콘텐츠 높이 계산
        total_h = pad + title_h + subtitle_h + divider_h
        for _, _, grouped in sections:
            total_h += section_header_h
            total_h += class_row_h * (len(grouped) if grouped else 1)
            total_h += section_gap
        total_h += pad
        panel_w = max(110.0, 168.0 * s)
        panel_round = max(5.0, 10.0 * s)

        # 패널 배경
        panel_rect = skia.RRect.MakeRectXY(
            skia.Rect(x0, y0, x0 + panel_w, y0 + total_h), panel_round, panel_round)
        canvas.drawRRect(panel_rect, self.hud_bg_paint)
        canvas.drawRRect(panel_rect, self.hud_border_paint)

        cx, cy = x0 + pad, y0 + pad

        # 텍스트 baseline 오프셋 — 폰트와 같이 스케일
        title_off = max(10.0, 16.0 * s)
        sub_off   = max(8.0,  13.0 * s)
        row_off   = max(8.0,  14.0 * s)
        ind_off_x = max(6.0,  10.0 * s)
        ind_r     = max(2.0,  4.0  * s)
        cls_off_x = max(12.0, 20.0 * s)
        sep_w     = max(0.5,  1.0  * s)

        # 타이틀
        title_paint = skia.Paint(AntiAlias=True)
        title_paint.setColor(skia.Color(230, 240, 255))
        canvas.drawString(hud_title, cx, cy + title_off, self.font_title, title_paint)
        cy += title_h

        # 부제 (비교 모드 모델명)
        if panel_title:
            sub_paint = skia.Paint(AntiAlias=True)
            sub_paint.setColor(skia.Color(170, 200, 240))
            canvas.drawString(panel_title, cx, cy + sub_off, self.font_lane, sub_paint)
            cy += subtitle_h

        # 구분선
        sep_paint = skia.Paint(AntiAlias=True)
        sep_paint.setColor(skia.Color(100, 120, 160))
        sep_paint.setAlphaf(0.3)
        sep_paint.setStrokeWidth(sep_w)
        canvas.drawLine(cx, cy, x0 + panel_w - pad, cy, sep_paint)
        cy += divider_h

        right_edge = x0 + panel_w - pad

        for section_name, total, non_zero in sections:
            # 섹션 헤더 + 합계
            lane_paint = skia.Paint(AntiAlias=True)
            lane_paint.setColor(skia.Color(140, 180, 255))
            canvas.drawString(f"{section_name}", cx, cy + row_off, self.font_lane, lane_paint)

            total_str = str(total)
            total_w = self.font_lane.measureText(total_str)
            total_paint = skia.Paint(AntiAlias=True)
            total_paint.setColor(skia.Color(255, 255, 255))
            canvas.drawString(total_str, right_edge - total_w, cy + row_off, self.font_lane, total_paint)
            cy += section_header_h

            if non_zero:
                for cls_display, count, (r, g, b) in non_zero:
                    ind_paint = skia.Paint(AntiAlias=True)
                    ind_paint.setColor(skia.Color(r, g, b))
                    canvas.drawCircle(cx + ind_off_x, cy + class_row_h / 2, ind_r, ind_paint)
                    cls_paint = skia.Paint(AntiAlias=True)
                    cls_paint.setColor(skia.Color(200, 210, 230))
                    canvas.drawString(cls_display, cx + cls_off_x, cy + row_off, self.font_count, cls_paint)

                    count_str = str(count)
                    count_w = self.font_count.measureText(count_str)
                    count_paint = skia.Paint(AntiAlias=True)
                    count_paint.setColor(skia.Color(255, 255, 255))
                    canvas.drawString(count_str, right_edge - count_w, cy + row_off, self.font_count, count_paint)
                    cy += class_row_h
            else:
                empty_paint = skia.Paint(AntiAlias=True)
                empty_paint.setColor(skia.Color(100, 110, 130))
                canvas.drawString("  -", cx + ind_off_x, cy + row_off, self.font_count, empty_paint)
                cy += class_row_h

            cy += section_gap


# ═══════════════════════════════════════════════════════════════
# 비디오 수집 / 첫 프레임 / ROI & 라인 해결
# ═══════════════════════════════════════════════════════════════

def _detect_headless():
    """HEADLESS 글로벌을 실제 bool 로 해석. True 면 GUI 호출(drawing) 차단."""
    if HEADLESS is True:
        return True
    if HEADLESS is False:
        return False
    # "auto"
    env = os.environ.get("HEADLESS", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    # Linux: DISPLAY / WAYLAND_DISPLAY 모두 비어있으면 헤드리스로 간주
    if sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            return True
    return False


def collect_video_paths():
    """VIDEO_PATHS (명시 리스트) → VIDEO_DIR+VIDEO_GLOB (패턴) 순으로 영상 경로 수집."""
    paths = []
    if VIDEO_PATHS:
        for p in VIDEO_PATHS:
            pp = Path(p)
            if pp.exists():
                paths.append(pp)
            else:
                print(f"  [warn] VIDEO_PATHS에 지정된 파일 없음: {p}")
    elif VIDEO_DIR:
        vd = Path(VIDEO_DIR)
        if vd.is_dir():
            paths = sorted(vd.glob(VIDEO_GLOB))
        else:
            print(f"  [warn] VIDEO_DIR이 디렉토리가 아님: {VIDEO_DIR}")

    if not paths:
        print("  [error] 처리할 영상이 없습니다. VIDEO_PATHS 또는 VIDEO_DIR/VIDEO_GLOB 설정 확인.")
    return paths


def read_first_frame(video_path):
    """영상의 첫 프레임을 읽어 반환. (frame, (w, h)) 또는 (None, (0, 0))."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [error] 영상을 열 수 없음: {video_path}")
        return None, (0, 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"  [error] 첫 프레임 읽기 실패: {video_path}")
        return None, (w, h)
    return frame, (w, h)


def _polylines_to_counting_lines(polylines_lst):
    """폴리라인 dict 리스트를 렌더링/카운팅용 [(id, coords_tuple), ...] 로 변환.
    좌표 스케일링은 이미 끝났다고 가정."""
    out = []
    for item in polylines_lst:
        coords = tuple(tuple(map(int, c)) for c in item["coords"])
        out.append((int(item["id"]), coords))
    return out


def resolve_roi(frame, wh, counting_lines, prev_roi=None):
    """ROI_MODE 에 따라 ROI 폴리곤 반환 (np.array shape (N,2)) 또는 None.

    모드:
      draw : 인터랙티브 그리기 (prev_roi를 편집 상태로 로드 가능)
      auto : counting_lines bbox + ROI_PADDING 사각형
    """
    if not ROI_ENABLED:
        return None

    w, h = wh

    if ROI_MODE == "draw":
        if frame is None:
            print("  [ROI] 프레임이 없어 draw 모드 스킵")
            return None
        if prev_roi is not None and len(prev_roi) > 0:
            print(f"  [ROI] Previous ROI loaded ({len(prev_roi)} points)")
        print("  [ROI] Interactive ROI drawing mode")
        roi = draw_roi_interactive(frame, counting_lines, prev_polygon=prev_roi)
        if roi is not None:
            print(f"  [ROI] Confirmed {len(roi)} points")
        else:
            print("  [ROI] Skipped - no ROI applied")
        return roi

    if ROI_MODE == "auto":
        if not counting_lines:
            print("  [ROI] auto 모드에 필요한 라인이 없어 스킵")
            return None
        all_xs, all_ys = [], []
        for _lid, coords in counting_lines:
            for x1s, y1s, x2s, y2s in coords:
                all_xs.extend([x1s, x2s])
                all_ys.extend([y1s, y2s])
        rx1 = max(0, min(all_xs) - ROI_PADDING)
        ry1 = max(0, min(all_ys) - ROI_PADDING)
        rx2 = min(w, max(all_xs) + ROI_PADDING)
        ry2 = min(h, max(all_ys) + ROI_PADDING)
        roi = np.array([[rx1, ry1], [rx2, ry1], [rx2, ry2], [rx1, ry2]], dtype=np.int32)
        print(f"  [ROI] 자동 계산 (카운팅 라인 bbox + {ROI_PADDING}px padding)")
        return roi

    print(f"  [ROI] 알 수 없는 ROI_MODE={ROI_MODE!r}")
    return None


# ═══════════════════════════════════════════════════════════════
# 그리기 페이즈
# ═══════════════════════════════════════════════════════════════

def drawing_phase(video_paths):
    """각 영상의 첫 프레임에서 라인 → ROI 순으로 그리고 per-video config 저장."""
    if _detect_headless():
        # 안전장치: 디스플레이가 없는 환경에서 우회 호출돼도 cv2.imshow 에서 죽지 않게 차단
        raise RuntimeError(
            "drawing_phase 는 디스플레이(imshow)가 필요합니다. "
            "디스플레이가 없는 환경(예: SSH+X11 미설정 서버)에서는 RUN_MODE='process_only' 로 실행하거나, "
            "GUI 가 동작하는 PC 에서 RUN_MODE='draw_only' 로 configs/*.json 을 먼저 생성한 뒤 "
            "input/ + configs/ + models/ 만 서버로 복사해 처리하세요."
        )
    print()
    print("=" * 60)
    print(f"  Drawing phase ({len(video_paths)} videos)")
    print("=" * 60)

    for vi, vp in enumerate(video_paths, 1):
        cfg_path = Path(CONFIG_DIR) / f"{vp.stem}.json"
        existing = load_video_config(cfg_path)

        print(f"\n  [{vi}/{len(video_paths)}] {vp.name}")

        if existing and OVERWRITE_EXISTING_CONFIG == "skip":
            print(f"    [skip] config exists: {cfg_path}")
            continue

        first_frame, (w, h) = read_first_frame(vp)
        if first_frame is None:
            print(f"    [skip] 첫 프레임 읽기 실패")
            continue

        # 편집 모드: 기존 config에서 이전 값 주입
        prev_lines = None
        prev_roi = None
        if existing and OVERWRITE_EXISTING_CONFIG == "edit":
            if existing.get("polylines_lst"):
                canvas = existing.get("canvas_size", f"{w}x{h}")
                src_w, src_h = map(int, canvas.split("x"))
                scaled = scale_lines(existing["polylines_lst"], (src_w, src_h), (w, h))
                prev_lines = [
                    {"id": lid, "num_lines": len(coords), "coords": [list(c) for c in coords]}
                    for lid, coords in scaled
                ]
            if existing.get("roi_polygon"):
                prev_roi = np.array(existing["roi_polygon"], dtype=np.int32)

        # ── 라인 먼저 (인터랙티브 그리기 / 편집) ──
        print("  [lines] Interactive line drawing mode")
        lines = draw_lines_interactive(first_frame, roi_polygon=prev_roi, prev_lines=prev_lines)
        if lines is None:
            print("    [skip] 라인 그리기 취소됨")
            continue
        print(f"    [lines] {len(lines)}개 확정")

        # ── ROI 나중 (방금 그린 lines를 참조선으로 사용) ──
        counting_lines_preview = _polylines_to_counting_lines(lines)
        roi = resolve_roi(first_frame, (w, h), counting_lines_preview, prev_roi=prev_roi)

        save_video_config(cfg_path, vp.stem, (w, h), lines, roi)


# ═══════════════════════════════════════════════════════════════
# 처리 페이즈
# ═══════════════════════════════════════════════════════════════

def _make_tracker_factory():
    """BoTSORT(ReID OFF) 팩토리 + 공유 device 리턴.

    ReID 는 사용하지 않음 (with_reid=False 고정). boxmot 의 BoTSORT 는
    with_reid=False 일 때 ReidAutoBackend 를 아예 초기화하지 않으므로
    model_weights 인자는 읽히지 않지만, 생성자 시그니처상 positional 필수
    인자라서 의미 없는 더미 Path 를 넘긴다. 이 경로는 절대 로드되지 않는다.
    """
    from boxmot.trackers.botsort.bot_sort import BoTSORT
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _UNUSED_REID_WEIGHTS = Path("__reid_disabled__")

    def _factory():
        return BoTSORT(
            model_weights=_UNUSED_REID_WEIGHTS,
            device=device,
            fp16=True,
            per_class=False,
            track_buffer=45,
            with_reid=False,
        )

    return _factory, device


# ═══════════════════════════════════════════════════════════════
# 모델 경로 resolver + 클래스 info YAML 로더
# ═══════════════════════════════════════════════════════════════

def load_class_info(info_path):
    """info-cls_*.yaml 을 {id:int -> name:str} dict 로 로드.

    지원하는 YAML 형식:
      (a) list: `names: ["car", "bus_s", ...]`  → index 를 id 로 사용
      (b) dict: `names: {0: "person", 2: "car", ...}` → 그대로 사용

    반환되는 이름은 원본 YAML 문자열 그대로 (언더스코어 표기를 유지).
    하이픈 표기로 섞여 들어와도 호출측에서 정규화 가능하도록
    원형 그대로 보존한다.
    """
    import yaml
    info_path = Path(info_path)
    if not info_path.exists():
        raise FileNotFoundError(f"Class info yaml not found: {info_path}")
    with open(info_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "names" not in data:
        raise ValueError(f"Invalid class info yaml (missing 'names'): {info_path}")
    names = data["names"]
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    raise TypeError(f"'names' must be list or dict in {info_path}, got {type(names).__name__}")


def derive_allowed_class_ids(label_mapping, allowed_names):
    """label_mapping ({id: name}) 에서 allowed_names 에 속한 이름의 id 를 역탐색.

    - allowed_names 가 None 이면 (None, None) 반환 → 전체 허용
    - 이름이 매칭되지 않으면 경고 출력 후 매칭된 것만 사용

    Returns
    -------
    (allowed_ids_set, filtered_label_mapping) | (None, None)
        filtered_label_mapping 은 allowed_ids 만 남긴 dict.
    """
    if allowed_names is None:
        return None, None
    wanted = set(allowed_names)
    matched = {cid: name for cid, name in label_mapping.items() if name in wanted}
    missing = wanted - set(matched.values())
    if missing:
        print(f"    [warn] allowed_names 중 YAML 에서 매칭되지 않은 이름: {sorted(missing)}")
    return set(matched.keys()), matched


def merge_stage1_and_classifier_label_maps(stage1_map, clf_map, offset=CLASSIFIER_STAGE2_ID_OFFSET):
    """Stage-1 {id:name} 와 분류기 {id:name} 를 단일 label_mapping 으로 병합.

    분류기 클래스 id 는 offset, offset+1, ... 에 대응.
    """
    out = dict(stage1_map)
    for i, name in clf_map.items():
        out[int(offset) + int(i)] = name
    return out


def _stage1_should_run_classifier(cls_name, apply_to_names):
    """재분류를 적용할 Stage-1 클래스명인지."""
    if cls_name in HIDDEN_CLASSES:
        return False
    if apply_to_names is not None:
        return cls_name in apply_to_names
    return normalize_class(cls_name) in ("car", "bus", "truck")


def load_classifier_bundle(key, device):
    """CLASSIFIERS 레지스트리에서 timm 분류기 로드 + 추론용 transform.

    Returns
    -------
    dict — keys: model, transform, clf_label_mapping, offset, min_bbox, apply_when,
            apply_to_names (frozenset|None), device, use_amp
    """
    import torch
    import timm
    import timm.data

    if key is None or key not in CLASSIFIERS:
        raise KeyError(f"Unknown classifier key: {key!r}")

    cfg = CLASSIFIERS[key]
    wpath = Path(cfg["path"])
    if not wpath.is_file():
        raise FileNotFoundError(f"Classifier weights not found: {wpath}")

    info = load_class_info(cfg["info"])
    timm_name = cfg.get("timm_model", "tf_efficientnet_b3_ns")
    model = timm.create_model(timm_name, pretrained=False, num_classes=len(info))

    ckpt = torch.load(wpath, map_location="cpu", weights_only=False)
    # 다양한 학습 스크립트 컨벤션을 지원: model_state_dict / state_dict / model / raw
    if isinstance(ckpt, dict):
        for cand in ("model_state_dict", "state_dict", "model"):
            if cand in ckpt and isinstance(ckpt[cand], dict):
                sd = ckpt[cand]
                break
        else:
            sd = ckpt
    else:
        sd = ckpt
    # 일부 학습 코드는 nn.Module 을 한 번 더 감싸서 'model.' prefix 가 붙는다 → 제거
    clean = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}
    model.load_state_dict(clean, strict=True)
    model.eval()
    model.to(device)

    # ── 추론 transform 결정 ──
    #   "raw"         : Resize((sz,sz)) + ToTensor (Normalize 없음).
    #                   12종 분류기 (production 학습 코드) 와 동일.
    #   "resize_norm" : Resize((sz,sz)) + ToTensor + Normalize(mean,std).
    #                   8종 분류기 (admin_storage 학습 코드) 와 동일. mean/std 는
    #                   normalize_mean / normalize_std 옵션 (기본 ImageNet).
    #   "timm"        : timm.data.create_transform 표준. timm 표준 fine-tune 에만.
    style = str(cfg.get("transform_style", "timm")).lower()
    sz = int(cfg.get("input_size", 300))
    if style == "raw":
        from torchvision import transforms as _tvt
        transform = _tvt.Compose([
            _tvt.Resize((sz, sz)),
            _tvt.ToTensor(),
        ])
        transform_desc = f"raw (Resize({sz},{sz}) + ToTensor; Normalize 없음 — 학습 코드와 동일)"
    elif style == "resize_norm":
        from torchvision import transforms as _tvt
        nmean = tuple(cfg.get("normalize_mean", [0.485, 0.456, 0.406]))
        nstd  = tuple(cfg.get("normalize_std",  [0.229, 0.224, 0.225]))
        transform = _tvt.Compose([
            _tvt.Resize((sz, sz)),
            _tvt.ToTensor(),
            _tvt.Normalize(mean=list(nmean), std=list(nstd)),
        ])
        transform_desc = (
            f"resize_norm (Resize({sz},{sz}) 비율 무시 강제 + ToTensor + "
            f"Normalize(mean={tuple(round(m, 3) for m in nmean)}, "
            f"std={tuple(round(s, 3) for s in nstd)}) — 학습 코드와 동일)"
        )
    else:
        data_config = timm.data.resolve_model_data_config(model)
        data_config.setdefault("input_size", (3, sz, sz))
        transform = timm.data.create_transform(**data_config, is_training=False)
        transform_desc = (
            f"timm default (Resize→CenterCrop({sz})→ToTensor→Normalize"
            f"(mean={tuple(round(m, 3) for m in data_config.get('mean', ()))}, "
            f"std={tuple(round(s, 3) for s in data_config.get('std', ()))}))"
        )

    apply_names = cfg.get("apply_to_names")
    if apply_names is not None:
        apply_names = frozenset(apply_names)

    use_amp = device.type == "cuda"

    print(f"  [2-stage] timm classifier '{key}' loaded: {timm_name}, {len(info)} classes, device={device}")
    print(f"    weights      : {wpath}")
    print(f"    info         : {cfg['info']}")
    print(f"    apply_when   : {cfg.get('apply_when', 'after_tracking')}")
    print(f"    transform    : {transform_desc}")
    print(f"    bbox_padding : {cfg.get('bbox_padding', STAGE2_BBOX_PADDING)}")

    return {
        "key"               : key,
        "model"             : model,
        "transform"         : transform,
        "clf_label_mapping" : info,
        "offset"            : CLASSIFIER_STAGE2_ID_OFFSET,
        "min_bbox"          : int(cfg.get("min_bbox", 32)),
        "apply_when"        : cfg.get("apply_when", "after_tracking"),
        "apply_to_names"    : apply_names,
        "device"            : device,
        "use_amp"           : use_amp,
        # 분류기별 padding 비율 (CLASSIFIERS 에서 지정 시 그 값, 없으면 전역 default)
        "bbox_padding"      : float(cfg.get("bbox_padding", STAGE2_BBOX_PADDING)),
    }


class TrackStage2Smoother:
    """track id 별 Stage-2 softmax 확률의 EMA 누적기.

    매 프레임 새 관측이 들어오면 기존 EMA 와 가중평균 후 그 EMA 의 argmax 를
    안정적인 클래스로 사용. 같은 track 의 분류 결과가 시간축으로 부드럽게
    수렴하므로 jitter 가 사라진다. 또한:
      - 신뢰도 (top-1, top1-top2 margin) 가 낮은 프레임은 갱신 자체를 skip
      - 일정 시간 보이지 않는 track 의 상태는 만료시켜 메모리 누수 방지
      - **Trust mode**: bbox 가 충분히 크고 (close-up) 분류기 confidence 도
        충분히 높은 "고품질 관측" 은 base alpha 대신 더 큰 trust_alpha 로
        EMA 를 갱신해, far 단계에서 잘못된 클래스에 잠긴 EMA 를 빠르게
        교정한다 ("객체가 가까워져서 비로소 정답 분류" 케이스 방어).
    """

    def __init__(self, num_classes, alpha=0.35,
                 min_top1=0.0, min_margin=0.0, ttl_frames=240,
                 trust_bbox_px=0, trust_top1=0.0, trust_alpha=0.0,
                 trust_adaptive=False, trust_pct=75.0,
                 trust_warmup=50, trust_floor_px=0,
                 trust_recompute_every=30, trust_max_samples=2000):
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.min_top1 = float(min_top1)
        self.min_margin = float(min_margin)
        self.ttl = int(ttl_frames)
        # trust mode params (trust_alpha <= alpha 또는 trust_bbox_px<=0 이면 비활성)
        # trust_bbox_px 는 fixed 값 또는 adaptive 의 fallback / warmup 값으로 사용.
        self.trust_bbox_px_fixed = float(trust_bbox_px)
        self.trust_top1          = float(trust_top1)
        self.trust_alpha         = float(trust_alpha)
        # adaptive trust threshold (video-level percentile 자동 조정)
        self._trust_adaptive       = bool(trust_adaptive)
        self._trust_pct            = float(trust_pct)
        self._trust_warmup         = int(trust_warmup)
        self._trust_floor          = float(trust_floor_px)
        self._trust_recompute_every = max(1, int(trust_recompute_every))
        if self._trust_adaptive:
            from collections import deque
            self._bbox_samples = deque(maxlen=int(trust_max_samples))
            self._cached_trust_threshold = self.trust_bbox_px_fixed
            self._obs_since_recompute = 0
            self._sample_dirty = True
        # tid -> np.ndarray [num_classes]
        self._probs = {}
        self._last_seen = {}

    def _current_trust_threshold(self):
        """현재 video 의 trust mode bbox 임계 (px). adaptive 면 percentile 사용."""
        if not self._trust_adaptive:
            return self.trust_bbox_px_fixed
        if len(self._bbox_samples) < self._trust_warmup:
            # 표본 부족 — fixed 값으로 폴백
            return self.trust_bbox_px_fixed
        if self._sample_dirty:
            arr = np.fromiter(self._bbox_samples, dtype=np.float32)
            v = float(np.percentile(arr, self._trust_pct))
            self._cached_trust_threshold = max(self._trust_floor, v)
            self._sample_dirty = False
            self._obs_since_recompute = 0
        return self._cached_trust_threshold

    def _trust_active(self, bbox_min_side, top1):
        """현재 관측이 trust mode (close-up + high conf) 자격을 갖췄는지."""
        if self.trust_alpha <= self.alpha:
            return False
        if bbox_min_side is None:
            return False
        threshold = self._current_trust_threshold()
        if threshold <= 0:
            return False
        return (bbox_min_side >= threshold) and (top1 >= self.trust_top1)

    def _record_bbox_observation(self, bbox_min_side):
        """adaptive 모드에서 percentile 추정용 표본 누적."""
        if not self._trust_adaptive:
            return
        if bbox_min_side is None or bbox_min_side <= 0:
            return
        self._bbox_samples.append(float(bbox_min_side))
        self._obs_since_recompute += 1
        if self._obs_since_recompute >= self._trust_recompute_every:
            self._sample_dirty = True

    def peek_class(self, tid, frame_idx):
        """현재 누적된 EMA 의 argmax 만 조회 — EMA 갱신 없음, TTL 만 연장.

        detection 누락으로 분류기 입력이 없는 프레임 (예: tracker 의 Kalman
        예측 bbox) 에서도 직전까지 누적된 안정 클래스를 시각화/카운팅에
        사용할 수 있게 해주는 read-only 조회 헬퍼.

        Returns
        -------
        int | None : EMA state 가 있으면 argmax index, 없으면 None.
        """
        if tid is None or int(tid) < 0:
            return None
        tid = int(tid)
        if tid in self._last_seen:
            self._last_seen[tid] = frame_idx
        cur = self._probs.get(tid)
        return None if cur is None else int(np.argmax(cur))

    def adaptive_threshold_snapshot(self):
        """현재 adaptive trust 임계값과 표본 수를 반환 (디버깅용).

        Returns
        -------
        dict | None : adaptive 비활성이면 None, 그 외엔
                      {"adaptive": True, "threshold_px": float,
                       "n_samples": int, "warmed_up": bool}
        """
        if not self._trust_adaptive:
            return None
        return {
            "adaptive"    : True,
            "threshold_px": float(self._current_trust_threshold()),
            "n_samples"   : int(len(self._bbox_samples)),
            "warmed_up"   : len(self._bbox_samples) >= self._trust_warmup,
        }

    def step(self, tid, probs, frame_idx, bbox_min_side=None):
        """probs: np.ndarray [num_classes] softmax. None 또는 저신뢰면 갱신 안함.

        bbox_min_side: 현재 bbox 의 min(width, height) [px]. trust mode 판정용.
                       전달 안 하면 trust mode 는 항상 OFF (base alpha 만 사용).

        Returns
        -------
        argmax index (int) of the smoothed distribution, or None if no state yet.
        """
        # adaptive trust mode: 모든 관측 (untracked 포함) 의 bbox 분포를 모음
        # → percentile 추정의 표본 풀에 가능한 많은 데이터 확보
        if self._trust_adaptive and bbox_min_side is not None:
            self._record_bbox_observation(bbox_min_side)

        if tid is None or int(tid) < 0:
            # 미트랙 검출은 EMA 의미가 없으므로 단발 argmax
            if probs is None:
                return None
            return int(np.argmax(probs))

        tid = int(tid)
        accept = probs is not None

        # ── top-1, top1-top2 margin 계산 (가드 + trust 판정 양쪽에 사용) ──
        top1 = 0.0
        margin = 0.0
        if probs is not None:
            top2_idx = np.argpartition(probs, -2)[-2:]
            a = float(probs[top2_idx[0]])
            b = float(probs[top2_idx[1]])
            top1 = a if a >= b else b
            margin = abs(a - b)

        if accept and (self.min_top1 > 0.0 or self.min_margin > 0.0):
            if top1 < self.min_top1 or margin < self.min_margin:
                accept = False

        if accept:
            # trust mode 면 더 큰 alpha 사용 (close-up + 확신 시 catch-up 가속)
            eff_alpha = self.trust_alpha if self._trust_active(bbox_min_side, top1) else self.alpha
            if tid in self._probs:
                self._probs[tid] = (1.0 - eff_alpha) * self._probs[tid] + eff_alpha * probs
            else:
                self._probs[tid] = probs.astype(np.float32, copy=True)
            self._last_seen[tid] = frame_idx
        else:
            # 갱신은 안 하지만 본 사실은 기록해 TTL 갱신
            if tid in self._last_seen:
                self._last_seen[tid] = frame_idx

        cur = self._probs.get(tid)
        return None if cur is None else int(np.argmax(cur))

    def cleanup(self, frame_idx):
        if not self._last_seen:
            return
        thr = frame_idx - self.ttl
        stale = [tid for tid, t in self._last_seen.items() if t < thr]
        for tid in stale:
            self._probs.pop(tid, None)
            self._last_seen.pop(tid, None)


def _crop_with_padding(frame_bgr, x1, y1, x2, y2, pad_ratio):
    """bbox 를 pad_ratio 만큼 확장하여 frame 경계 안에서 crop. (x1c..y2c, crop)."""
    h, w = frame_bgr.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    if pad_ratio > 0 and bw > 0 and bh > 0:
        dx = int(round(bw * pad_ratio))
        dy = int(round(bh * pad_ratio))
        x1 -= dx; x2 += dx
        y1 -= dy; y2 += dy
    x1c = max(0, min(int(x1), w - 1))
    y1c = max(0, min(int(y1), h - 1))
    x2c = max(x1c + 1, min(int(x2), w))
    y2c = max(y1c + 1, min(int(y2), h))
    return x1c, y1c, x2c, y2c, frame_bgr[y1c:y2c, x1c:x2c]


def apply_stage2_classification(
    frame_bgr,
    boxes_xy6,
    stage1_label_mapping,
    bundle,
    smoother=None,
    frame_idx=0,
):
    """boxes [N,6] 의 cls 컬럼을 Stage-2 분류 결과로 덮어쓴다 (복사본 반환).

    boxes: float or int array [x1,y1,x2,y2,tid,cls] — cls 는 Stage-1 id.

    개선점:
      1) 한 프레임 안 모든 분류 대상 detection 을 모아 batch forward (속도)
      2) softmax 후 (smoother 있으면) track id 별 EMA 로 안정화 (jitter 제거)
      3) bbox 패딩 추가 (분류 정확도 향상)
      4) 저신뢰 프레임은 EMA 갱신 skip (smoother 안에서 처리)
    """
    import torch

    if boxes_xy6 is None or len(boxes_xy6) == 0:
        return boxes_xy6

    out = boxes_xy6.copy()
    model = bundle["model"]
    transform = bundle["transform"]
    device = bundle["device"]
    offset = bundle["offset"]
    min_side = bundle["min_bbox"]
    apply_to_names = bundle["apply_to_names"]
    # 분류기별 padding 비율 (CLASSIFIERS 의 bbox_padding) 우선 사용. 없으면 전역 default.
    pad_ratio = float(bundle.get("bbox_padding", STAGE2_BBOX_PADDING))
    use_batch = bool(STAGE2_BATCH_INFER)

    # ── 1) 분류 대상 인덱스 + 입력 텐서 수집 ──
    #   설계 원칙: "라벨은 무조건 12종, EMA 만 신뢰입력으로 가드"
    #     - HIDDEN(person/bike) 또는 _stage1_should_run_classifier 통과 못 한 행:
    #       분류 대상 자체가 아니므로 stage-2 건드리지 않음 (raw 라벨 유지).
    #     - frame 밖이라 crop 이 빈 행: inference 불가 → peek_only 로 EMA 가
    #       있으면 라벨 유지, 없으면 stage-1 fallback.
    #     - 그 외 (작은 bbox 포함) 모든 차량 행: 무조건 분류기에 보내서 12종
    #       라벨을 출력. 단 EMA 갱신은 step() 안의 신뢰도 가드 + bbox_min_side
    #       기준으로 분류기에 보낸 행 중에서 다시 한 번 걸러진다.
    #     - update_ema 플래그가 False 인 행 (작은 bbox) 은 분류기 단발 결과
    #       또는 직전 EMA stable class 를 라벨로 사용하지만 EMA 누적에는 영향
    #       없음 → 작은 bbox 의 부정확한 출력이 평균을 오염시키지 않음.
    from PIL import Image
    indices = []          # out 의 행 인덱스 (분류기에 보낸 모든 행)
    tensors = []          # transform 적용된 텐서들
    bbox_sizes = []       # 각 분류 대상의 bbox min(w,h)
    update_ema = []       # 행마다 EMA 갱신을 허용할지 (True/False)
    peek_only_indices = []  # 분류기 inference 자체를 못 한 행 (frame 밖 등)

    # is_predicted (마지막 컬럼) = 1 인 행은 Kalman 예측 bbox → 분류 입력으로
    # 신뢰할 수 없으므로 분류기/EMA/percentile 모두 건드리지 않음. (4) 단계에서
    # peek_class 로 EMA stable class 만 덮어쓴다.
    has_pred_flag = (out.ndim == 2 and out.shape[1] >= 7)

    for i in range(len(out)):
        if has_pred_flag and int(out[i, 6]) == 1:
            continue
        sid = int(out[i, 5])
        raw_name = stage1_label_mapping.get(sid)
        if raw_name is None:
            continue
        # HIDDEN (person/bike 등) 은 stage-2 대상 아님 → 라벨도 건드리지 않음.
        if raw_name in HIDDEN_CLASSES:
            continue
        # apply_to_names 또는 normalize_class 필터에서 빠진 클래스: stage-2 안
        # 돌리지만 EMA 가 있다면 stable class 로 라벨 유지 (예: 같은 track 이
        # 한 frame 만 motorcycle 로 잘못 잡혔을 때 직전 12종 라벨 유지).
        if not _stage1_should_run_classifier(raw_name, apply_to_names):
            peek_only_indices.append(i)
            continue

        x1, y1, x2, y2 = map(int, out[i, :4])
        bw, bh = x2 - x1, y2 - y1
        if bw < 1 or bh < 1:
            # 정상적인 bbox 가 아님 (degenerate) — inference 자체가 무의미
            peek_only_indices.append(i)
            continue

        _, _, _, _, crop = _crop_with_padding(frame_bgr, x1, y1, x2, y2, pad_ratio)
        if crop.size == 0:
            # frame 밖이라 실제 픽셀이 0개: inference 불가 → peek 으로 라벨 유지
            peek_only_indices.append(i)
            continue

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensors.append(transform(pil))
        indices.append(i)
        # 패딩 전 원본 bbox 크기 (분류기가 본 차량 자체 크기) — trust 판정용
        bbox_sizes.append(min(bw, bh))
        # EMA 갱신은 신뢰할 수 있는 큰 bbox 만 허용 (작은 bbox 의 단발 결과는
        # 라벨로는 쓰이지만 EMA 누적은 오염시키지 않음).
        update_ema.append(min(bw, bh) >= min_side)

    if not indices:
        # 분류 대상은 없지만, peek_only / Kalman 예측 행은 EMA 라벨로 덮어쓸 수 있음
        if smoother is not None:
            for i in peek_only_indices:
                stable = smoother.peek_class(int(out[i, 4]), frame_idx)
                if stable is not None:
                    out[i, 5] = offset + stable
            if has_pred_flag:
                for i in range(len(out)):
                    if int(out[i, 6]) != 1:
                        continue
                    stable = smoother.peek_class(int(out[i, 4]), frame_idx)
                    if stable is not None:
                        out[i, 5] = offset + stable
            smoother.cleanup(frame_idx)
        return out

    # ── 2) Forward (batch or one-by-one) ──
    with torch.no_grad():
        if use_batch:
            batch = torch.stack(tensors, dim=0).to(device)
            if bundle.get("use_amp"):
                with torch.amp.autocast("cuda"):
                    logits = model(batch)
            else:
                logits = model(batch)
            probs_t = torch.softmax(logits, dim=1)
            probs_np = probs_t.float().cpu().numpy()  # [B, C]
        else:
            probs_list = []
            for t in tensors:
                t = t.unsqueeze(0).to(device)
                if bundle.get("use_amp"):
                    with torch.amp.autocast("cuda"):
                        l = model(t)
                else:
                    l = model(t)
                probs_list.append(torch.softmax(l, dim=1).float().cpu().numpy()[0])
            probs_np = np.stack(probs_list, axis=0)

    # ── 3) Smoothing 후 cls 덮어쓰기 ──
    #   라벨 결정 우선순위 (분류기 결과는 항상 사용해서 12종 라벨 보장):
    #     update_ema=True  → smoother.step(...) 으로 EMA 누적 + stable_pred 사용
    #                        (저신뢰 가드 통과 못 하면 직전 EMA argmax 가 나옴.
    #                        그래도 None 이면 단발 argmax 로 fallback).
    #     update_ema=False → EMA 누적 X. 같은 track 의 직전 EMA 가 있으면 그
    #                        stable class 사용 (label 일관성 우선). 없으면 분류기
    #                        단발 argmax 사용 (작은 bbox 라도 12종 출력 보장).
    for k, i in enumerate(indices):
        probs = probs_np[k]
        if smoother is not None:
            tid = int(out[i, 4])
            if update_ema[k]:
                stable_pred = smoother.step(
                    tid, probs, frame_idx, bbox_min_side=bbox_sizes[k]
                )
                pred = stable_pred if stable_pred is not None else int(np.argmax(probs))
            else:
                stable_pred = smoother.peek_class(tid, frame_idx)
                pred = stable_pred if stable_pred is not None else int(np.argmax(probs))
        else:
            pred = int(np.argmax(probs))
        out[i, 5] = offset + pred

    # ── 4) stage-2 skip 행 (peek_only) + Kalman 예측 행 → EMA stable class 로 덮어쓰기 ──
    #   분류기는 안 돌렸지만 같은 track 의 직전 EMA 가 있으면 그 argmax 를
    #   사용해 12종 라벨을 유지한다. EMA 가 비면 (track 첫 등장이 skip 케이스
    #   였으면) stage-1 라벨이 그대로 노출되는데, 이건 본질적으로 fallback 이라
    #   피할 수 없음. 그러나 한 frame 이라도 stage-2 가 통과한 track 은 이후
    #   영원히 12종 라벨이 유지된다.
    if smoother is not None:
        for i in peek_only_indices:
            stable = smoother.peek_class(int(out[i, 4]), frame_idx)
            if stable is not None:
                out[i, 5] = offset + stable
        if has_pred_flag:
            for i in range(len(out)):
                if int(out[i, 6]) != 1:
                    continue
                stable = smoother.peek_class(int(out[i, 4]), frame_idx)
                if stable is not None:
                    out[i, 5] = offset + stable
        smoother.cleanup(frame_idx)
    return out


def apply_stage2_to_detections(frame_bgr, det, stage1_label_mapping, bundle,
                               smoother=None, frame_idx=0):
    """YOLO 검출 [N,6] xyxy,conf,cls 에 Stage-2 를 적용 (apply_when=after_detection).

    검출 단계에서는 track id 가 없으므로 smoother 효과가 제한적.
    """
    if det is None or len(det) == 0:
        return det
    fake_tid = np.full(len(det), -1, dtype=np.float32)
    six = np.column_stack([det[:, :4], fake_tid, det[:, 5]])
    six2 = apply_stage2_classification(
        frame_bgr, six, stage1_label_mapping, bundle,
        smoother=smoother, frame_idx=frame_idx,
    )
    out = det.copy()
    out[:, 5] = six2[:, 5]
    return out


def resolve_model_path(path_or_name):
    """Detection 모델 파일 경로를 DETECTION_DIR 기준으로 정규화.

    우선순위:
      1) 인자가 이미 존재하는 파일 경로이면 그대로 반환
      2) DETECTION_DIR/<파일명> 이 존재하면 그 경로 반환
      3) 둘 다 아니면 DETECTION_DIR/<파일명> 을 반환 (파일은 아직 없음).
         실제 다운로드는 이후 _load_yolo_detection_model() 에서 처리.
    """
    p = Path(path_or_name)
    if p.exists():
        return p
    dest = DETECTION_DIR / p.name
    if dest.exists():
        return dest
    return dest


def _load_yolo_detection_model(path_or_name):
    """YOLO detection 모델 로드 + 자동 다운로드된 파일을 DETECTION_DIR 로 이동.

    로컬에 파일이 없는 경우 ultralytics 는 현재 작업 디렉토리(CWD) 에
    가중치를 떨어뜨리는 기본 동작을 가진다. 이 함수는 로드 전후의 CWD
    .pt 파일 집합을 비교해 새로 생성된 파일을 DETECTION_DIR 로 옮긴다.
    다음 실행부터는 resolve_model_path 가 DETECTION_DIR 안에서 파일을 찾는다.

    Returns
    -------
    (YOLO, Path)
        로드된 모델 인스턴스와 최종적으로 확정된 모델 파일 경로.
    """
    target = resolve_model_path(path_or_name)
    if target.exists():
        return YOLO(str(target), task="detect"), target

    # 자동 다운로드 경로 — DETECTION_DIR 보장 + CWD 스냅샷
    DETECTION_DIR.mkdir(parents=True, exist_ok=True)
    before = {p.resolve() for p in Path(".").glob("*.pt")}
    model = YOLO(target.name, task="detect")   # 이름만 전달 → CWD 에 다운로드
    after = {p.resolve() for p in Path(".").glob("*.pt")}

    final_path = target
    for nf in after - before:
        dest = DETECTION_DIR / nf.name
        if dest.exists():
            # 이미 같은 이름이 DETECTION_DIR 에 있으면 중복 다운로드분은 제거
            nf.unlink()
        else:
            nf.rename(dest)
            print(f"    [resolve_model_path] auto-downloaded {nf.name} -> {dest}")
        if nf.name == target.name:
            final_path = dest

    return model, final_path


def _counter_state_snapshot(counter):
    """SimpleCounter 의 필드를 렌더러에 전달할 dict 로 패키징."""
    return {
        "lane_class_counts": counter.lane_class_counts,
        "line_class_counts": counter.line_class_counts,
        "total_class_counts": counter.total_class_counts,
        "line_unique_tracks": counter.line_unique_tracks,
    }


def _extract_tracked_bboxes(tracker, det, frame):
    """YOLO detection 을 tracker 에 넘기고 [N, 7] int array 로 정규화.

    Output columns: [x1, y1, x2, y2, track_id, cls_idx, is_predicted]
      - is_predicted = 0 : 이번 프레임에 detection 매칭된 정상 출력
      - is_predicted = 1 : detection 누락이지만 lost track 의 Kalman 예측 bbox로
                           gap 을 메운 행 (시각적 연속성용; 분류기는 skip)

    TRACKER_USE_LOST_PREDICTION 이 False 면 is_predicted=1 행은 추가하지 않음
    (모든 행이 is_predicted=0).
    """
    # det 가 비어 있어도 multi_predict 진행은 여전히 호출돼야 lost track 위치가
    # 갱신됨 → tracker.update 에 빈 detection 을 전달.
    tracked = tracker.update(det if det.shape[0] > 0 else np.empty((0, 6)), frame)

    # 1) detection 매칭된 활성 출력 정규화
    if tracked is None or len(tracked) == 0 or tracked.shape[0] == 0:
        tb_active = np.empty((0, 6), dtype=int)
    else:
        valid_mask = ~np.isnan(tracked).any(axis=1)
        tracked = tracked[valid_mask]
        if tracked.shape[0] == 0:
            tb_active = np.empty((0, 6), dtype=int)
        else:
            if tracked.shape[1] >= 7:
                tb_active = np.column_stack([tracked[:, :5], tracked[:, 6]]).astype(int)
            else:
                tb_active = tracked[:, :6].astype(int)
            tb_active[:, :4] = np.maximum(tb_active[:, :4], 0)

    # is_predicted=0 컬럼 추가
    if tb_active.shape[0] > 0:
        flag_col = np.zeros((tb_active.shape[0], 1), dtype=int)
        tb_out = np.hstack([tb_active, flag_col])
    else:
        tb_out = np.empty((0, 7), dtype=int)

    # 2) lost track 의 Kalman 예측 bbox 합치기 (옵션)
    if not TRACKER_USE_LOST_PREDICTION:
        return tb_out

    lost = getattr(tracker, "lost_stracks", None)
    if not lost:
        return tb_out

    cur_frame = int(getattr(tracker, "frame_count", 0))
    active_ids = set(tb_out[:, 4].astype(int).tolist()) if tb_out.shape[0] > 0 else set()
    extras = []
    H, W = (frame.shape[0], frame.shape[1]) if frame is not None else (None, None)

    for st in lost:
        tid = int(getattr(st, "id", -1))
        if tid in active_ids or tid < 0:
            continue
        gap = cur_frame - int(getattr(st, "frame_id", cur_frame))
        # gap=0 (방금 lost) ~ TRACKER_PREDICT_MAX_GAP 까지만 신뢰
        if gap <= 0 or gap > int(TRACKER_PREDICT_MAX_GAP):
            continue
        try:
            x1, y1, x2, y2 = st.xyxy
        except Exception:
            continue
        if not (np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2)):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        # 화면 밖이면 제외 (예측이 너무 멀리 drift 한 경우)
        if H is not None and (x2 < 0 or y2 < 0 or x1 >= W or y1 >= H):
            continue
        extras.append([int(x1), int(y1), int(x2), int(y2),
                       tid, int(getattr(st, "cls", 0)), 1])

    if extras:
        ext_arr = np.array(extras, dtype=int)
        ext_arr[:, :4] = np.maximum(ext_arr[:, :4], 0)
        tb_out = np.vstack([tb_out, ext_arr])

    return tb_out


def _run_yolo_detection(model, frame, allowed_classes):
    """YOLO 예측 후 [x1,y1,x2,y2,conf,cls] float array 반환.
    allowed_classes(set[int]) 가 주어지면 해당 class 만 남김."""
    results = model.predict(frame, verbose=False, conf=0.25, iou=0.45, imgsz=960)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return np.empty((0, 6))
    det = np.hstack((
        results.boxes.xyxy.cpu().numpy(),
        results.boxes.conf.cpu().numpy()[:, np.newaxis],
        results.boxes.cls.cpu().numpy()[:, np.newaxis],
    ))
    if allowed_classes is not None and det.shape[0] > 0:
        mask = np.isin(det[:, 5].astype(int), list(allowed_classes))
        det = det[mask]
    return det


def process_single_video(video_path, cfg, engines_spec, tracker_factory, batch_timestamp, classifier_bundle=None):
    """단일 영상에 대해 프레임 루프 + writer + ffmpeg + 요약.

    engines_spec: list of dict — 각 엔진별로:
        {
            "name"           : 로그용 짧은 이름 (예: "custom-8cls")
            "title"          : HUD 패널 상단 부제 (None 이면 미표시)
            "model"          : 로드된 ultralytics YOLO 모델
            "label_mapping"  : {cls_idx: "car" 등} — 해당 모델 전용 매핑
            "allowed_classes": set[int] | None — None 이면 모든 클래스 허용
        }
    len(engines_spec) > 1 이면 horizontal concat.
    classifier_bundle: ACTIVE_CLASSIFIER 활성 시 load_classifier_bundle() 결과.
        Stage-1 id 공간 + CLASSIFIER_STAGE2_ID_OFFSET 에 Stage-2 클래스를 합친 merged mapping 을
        counter/renderer 에 사용한다.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [error] 영상을 열 수 없음: {video_path}")
        return None

    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(MAX_SECONDS * vid_fps) if MAX_SECONDS is not None else total_frames
    max_frames = min(max_frames, total_frames)

    compare = len(engines_spec) > 1
    print(f"  영상: {video_path}")
    print(f"  해상도: {w}x{h} @ {vid_fps:.1f}fps  (engines={len(engines_spec)}{' compare' if compare else ''})")
    print(f"  처리 프레임: {max_frames} / {total_frames}")

    # ── 카운팅 라인 해결 (config 좌표 → 영상 해상도로 스케일) ──
    polylines_lst = cfg.get("polylines_lst") if cfg else None
    if not polylines_lst:
        print("  [error] config에 polylines_lst 없음 — 스킵")
        cap.release()
        return None

    canvas_sz = cfg.get("canvas_size", f"{w}x{h}")
    src_w, src_h = map(int, canvas_sz.split("x"))
    counting_lines = scale_lines(polylines_lst, (src_w, src_h), (w, h))
    print(f"  카운팅 라인 {len(counting_lines)}개 로드")

    # ── ROI (config가 유일한 소스) ──
    roi_polygon = None
    if ROI_ENABLED and cfg.get("roi_polygon"):
        roi_raw = np.array(cfg["roi_polygon"], dtype=np.float32)
        if src_w != w or src_h != h:
            roi_raw[:, 0] *= (w / src_w)
            roi_raw[:, 1] *= (h / src_h)
        roi_polygon = roi_raw.astype(np.int32)
        print(f"  [ROI] config에서 로드 ({len(roi_polygon)}점)")

    # ── 엔진별 tracker / counter / renderer 구축 ──
    engines = []
    for spec in engines_spec:
        s1map = spec["label_mapping"]
        use_s2 = bool(classifier_bundle) and bool(spec.get("stage2", False))
        if use_s2:
            merged = merge_stage1_and_classifier_label_maps(
                s1map,
                classifier_bundle["clf_label_mapping"],
                classifier_bundle["offset"],
            )
        else:
            merged = s1map
        # Stage-2 EMA smoother (track id 별 분류 안정화)
        smoother = None
        if use_s2 and STAGE2_SMOOTH_ENABLED:
            num_clf_classes = len(classifier_bundle["clf_label_mapping"])
            ttl_frames = max(1, int(STAGE2_SMOOTH_TTL_SEC * vid_fps))
            smoother = TrackStage2Smoother(
                num_classes=num_clf_classes,
                alpha=float(STAGE2_SMOOTH_ALPHA),
                min_top1=float(STAGE2_MIN_TOP1_PROB),
                min_margin=float(STAGE2_MIN_MARGIN),
                ttl_frames=ttl_frames,
                trust_bbox_px=int(STAGE2_TRUST_BBOX_PX),
                trust_top1=float(STAGE2_TRUST_TOP1),
                trust_alpha=float(STAGE2_TRUST_ALPHA),
                trust_adaptive=bool(STAGE2_TRUST_BBOX_ADAPTIVE),
                trust_pct=float(STAGE2_TRUST_BBOX_PCT),
                trust_warmup=int(STAGE2_TRUST_BBOX_WARMUP),
                trust_floor_px=float(STAGE2_TRUST_BBOX_FLOOR),
                trust_recompute_every=int(STAGE2_TRUST_BBOX_RECOMPUTE_EVERY),
            )

        eng = {
            "name"                : spec["name"],
            "title"               : spec.get("title"),
            "tag"                 : spec.get("tag", spec["name"]),
            "model"               : spec["model"],
            "label_mapping"       : merged,
            "stage1_label_mapping": s1map,
            "use_stage2"          : use_s2,
            "stage2_smoother"     : smoother,
            "allowed_classes"     : spec.get("allowed_classes"),
            "tracker"             : tracker_factory(),
            "counter"             : SimpleCounter(counting_lines, merged, target_fps=int(vid_fps)),
            "renderer"            : SkiaCountingRenderer(w, h, counting_lines, roi_polygon=roi_polygon),
        }
        engines.append(eng)

    # ── 비디오 라이터 ──
    # 파일명 규칙 (단일/비교 모두 공통 포맷, 모델 구분은 tag 영역으로만):
    #   counting_result_{video_stem}__{engine_part}[_cls-{key}]_{batch_timestamp}.{ext}
    #   engine_part : 단일은 "{tag}", 비교는 "{tag1}-vs-{tag2}-..." (tag 는 모델 stem 을 sanitize)
    # 즉 "_compare_" 리터럴 접미사는 없애고, '-vs-' 존재 여부로 단/비를 판별한다.
    engine_part = "-vs-".join(eng["tag"] for eng in engines)
    base_name = f"counting_result_{video_path.stem}__{engine_part}_{batch_timestamp}"
    if classifier_bundle is not None and any(eng.get("use_stage2") for eng in engines):
        ctag = _sanitize_filename_component(str(classifier_bundle["key"]), fallback="cls")
        base_name = f"counting_result_{video_path.stem}__{engine_part}__cls-{ctag}_{batch_timestamp}"
    tmp_avi_path = Path(OUTPUT_DIR) / f"{base_name}.avi"
    output_mp4_path = Path(OUTPUT_DIR) / f"{base_name}.mp4"
    out_w = w * len(engines) if compare else w
    writer = cv2.VideoWriter(
        str(tmp_avi_path),
        cv2.VideoWriter_fourcc(*"XVID"),
        vid_fps,
        (out_w, h),
    )

    # ── 프레임 처리 루프 ──
    clf_note = ""
    if classifier_bundle is not None and any(eng.get("use_stage2") for eng in engines):
        panels_s2 = [eng["name"] for eng in engines if eng.get("use_stage2")]
        clf_note = (f"  Stage-2 cls={classifier_bundle['key']}  apply_when={classifier_bundle['apply_when']}"
                    f"  panels={panels_s2}")
    print(f"  프레임 처리 시작... (COUNTING_MODE={COUNTING_MODE}){clf_note}")
    if classifier_bundle is not None and any(eng.get("use_stage2") for eng in engines):
        if classifier_bundle.get("apply_when") == "after_tracking" and not USE_TRACKER_BBOX:
            print("  [warn] USE_TRACKER_BBOX=False 이고 Stage-2 가 after_tracking 이면,"
                  " 시각화 박스가 Stage-1 라벨일 수 있음 — 2-stage 에서는 USE_TRACKER_BBOX=True 권장")
    perf = time.perf_counter
    t_start = perf()
    frame_idx = 0
    t_read = t_yolo = t_track = t_clf = t_count = t_render = t_write = 0.0

    for _ in _progress_iter(range(max_frames), max_frames, video_path.stem):
        t0 = perf()
        ret, frame = cap.read()
        if not ret:
            break
        t_read += perf() - t0

        video_time_ms = int(frame_idx / vid_fps * 1000)

        panels = []
        for eng in engines:
            # ── YOLO 검출 ──
            t0 = perf()
            det = _run_yolo_detection(eng["model"], frame, eng["allowed_classes"])
            t_yolo += perf() - t0

            # ── Stage-2 (검출 직후, 트래커 입력 전) — use_stage2 인 패널만 ──
            if (classifier_bundle is not None and eng.get("use_stage2")
                    and classifier_bundle.get("apply_when") == "after_detection"):
                t0 = perf()
                det = apply_stage2_to_detections(
                    frame, det, eng["stage1_label_mapping"], classifier_bundle,
                    smoother=eng.get("stage2_smoother"), frame_idx=frame_idx)
                t_clf += perf() - t0

            # ── 트래킹 ──
            t0 = perf()
            tracked_bboxes = _extract_tracked_bboxes(eng["tracker"], det, frame)
            t_track += perf() - t0

            # ── Stage-2 (트래킹 후, 카운팅/시각화 직전) — use_stage2 인 패널만 ──
            if (classifier_bundle is not None and eng.get("use_stage2")
                    and classifier_bundle.get("apply_when") == "after_tracking"):
                t0 = perf()
                tracked_bboxes = apply_stage2_classification(
                    frame, tracked_bboxes, eng["stage1_label_mapping"], classifier_bundle,
                    smoother=eng.get("stage2_smoother"), frame_idx=frame_idx)
                t_clf += perf() - t0

            # ── ROI 외부 박스 필터링 (hide 모드 전용) ──
            #   카운터/렌더러 호출 전에 한 번 잘라내어 외부에서 라인을 지나는
            #   객체가 카운트되지 않도록 하고 시각화도 깔끔하게 유지.
            tracked_bboxes = filter_boxes_by_roi(
                tracked_bboxes, eng["label_mapping"], roi_polygon)

            # ── 카운팅 ──
            t0 = perf()
            eng["counter"].update(tracked_bboxes, video_time_ms, frame_idx)
            t_count += perf() - t0

            # ── 시각화할 bbox 선택 ──
            if USE_TRACKER_BBOX:
                draw_boxes = tracked_bboxes
            else:
                if det.shape[0] > 0:
                    draw_boxes = np.column_stack([
                        det[:, :4].astype(int),
                        np.full(len(det), -1, dtype=int),
                        det[:, 5].astype(int),
                    ])
                    # detection 기반 시각화에도 동일한 ROI 필터 적용
                    draw_boxes = filter_boxes_by_roi(
                        draw_boxes, eng["label_mapping"], roi_polygon)
                else:
                    draw_boxes = np.empty((0, 6), dtype=int)

            # ── Skia 렌더링 ──
            t0 = perf()
            rendered = eng["renderer"].draw(
                frame, draw_boxes, eng["label_mapping"],
                _counter_state_snapshot(eng["counter"]),
                hit_events=eng["counter"].hit_events,
                frame_idx=frame_idx,
                panel_title=eng.get("title") if compare else None,
            )
            t_render += perf() - t0
            panels.append(rendered)

        combined = panels[0] if len(panels) == 1 else cv2.hconcat(panels)

        # ── 프레임 쓰기 ──
        t0 = perf()
        writer.write(combined)
        t_write += perf() - t0

        frame_idx += 1

    elapsed = perf() - t_start
    cap.release()
    writer.release()

    # ── 성능 요약 ──
    n = max(frame_idx, 1)
    print(f"\n{'=' * 60}")
    print(f"  {'성능 요약 - ' + video_path.stem:^52}")
    print(f"{'=' * 60}")
    print(f"  {'Step':<20} {'ms/frame':>10} {'total(s)':>10} {'비율':>8}")
    print(f"  {'-' * 50}")
    perf_rows = [
        ("Video read",   t_read),
        ("YOLO detect",  t_yolo),
        ("BoTSORT track", t_track),
    ]
    if classifier_bundle is not None:
        perf_rows.append(("Stage-2 clf", t_clf))
    perf_rows.extend([
        ("Counting",     t_count),
        ("Skia render",  t_render),
        ("Video write",  t_write),
    ])
    for name, t in perf_rows:
        print(f"  {name:<20} {t/n*1000:>10.1f} {t:>10.1f} {t/elapsed*100:>7.1f}%")
    print(f"  {'-' * 50}")
    print(f"  {'Total':<20} {elapsed/n*1000:>10.1f} {elapsed:>10.1f}")
    print(f"  처리: {frame_idx}프레임, {elapsed:.1f}초 ({n / elapsed:.1f} FPS)")
    print(f"{'=' * 60}")

    # ── AVI → MP4 변환 ──
    print("  영상 압축 중 (AVI -> MP4)...")
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-i", str(tmp_avi_path),
                "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
                "-y", str(output_mp4_path),
            ],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            os.remove(tmp_avi_path)
            print(f"  출력: {output_mp4_path}")
        else:
            print(f"  FFmpeg 오류: {result.stderr[:300]}")
    except FileNotFoundError:
        print("  ffmpeg 미설치. AVI 파일 유지됩니다.")
    except subprocess.TimeoutExpired:
        print("  FFmpeg 타임아웃")

    # ── 영상별 카운팅 요약 (엔진별로 출력) ──
    print()
    print("=" * 60)
    print(f"  카운팅 결과 - {video_path.stem}  (mode={COUNTING_MODE}"
          f"{', compare' if compare else ''})")
    print("=" * 60)
    video_total = 0
    per_engine = {}
    for eng in engines:
        counter = eng["counter"]
        tag = f"{eng['name']}"
        print(f"  [{tag}]")
        engine_total = 0
        if COUNTING_MODE == "single_line":
            # 라인별 고유 track 수 + 차종별 전체 합
            for line_idx, _ in counting_lines:
                label = SINGLE_LINE_LABELS.get(line_idx, f"Line {line_idx}")
                cnt = len(counter.line_unique_tracks.get(line_idx, ()))
                cls_counts = counter.line_class_counts.get(line_idx, {})
                grouped = group_class_counts(cls_counts)
                if grouped:
                    detail = ", ".join(f"{n}:{c}" for n, c, _ in grouped)
                    print(f"    {label}: {cnt}  [{detail}]")
                else:
                    print(f"    {label}: {cnt}")
            total_grouped = group_class_counts(counter.total_class_counts)
            engine_total = sum(c for _, c, _ in total_grouped)
            if total_grouped:
                detail = ", ".join(f"{n}:{c}" for n, c, _ in total_grouped)
                print(f"    Total: {engine_total}  [{detail}]")
            else:
                print(f"    Total: {engine_total}")
        else:
            for direction, lane_name in LANE_MAP.items():
                counts = counter.lane_class_counts.get(direction, {})
                grouped = group_class_counts(counts)
                total = sum(c for _, c, _ in grouped)
                engine_total += total
                if total > 0:
                    detail = ", ".join(f"{n}:{c}" for n, c, _ in grouped)
                    print(f"    {lane_name} ({direction}): {total}  [{detail}]")
                else:
                    print(f"    {lane_name} ({direction}): 0")
            print(f"    Total: {engine_total}")
        per_engine[eng["name"]] = engine_total
        video_total += engine_total
    print(f"  {'─' * 40}")
    print(f"  Video total (sum over engines): {video_total}")
    print("=" * 60)

    return {
        "video": video_path.name,
        "total": video_total,
        "per_engine": per_engine,
        "elapsed": elapsed,
        "frames": frame_idx,
    }


def _build_engines_spec():
    """ACTIVE_DETECTORS + DETECTORS 레지스트리에서 engines_spec 을 구성.

    각 detector 엔트리는 다음 과정으로 조립된다:
      1) path 로부터 YOLO 모델 로드 (필요 시 DETECTION_DIR 로 자동 이동)
      2) info YAML 에서 {id: name} label_mapping 로드 (Single source of truth)
      3) allowed_names 가 주어졌다면 YAML 에서 id 역탐색 → filtered label_mapping

    반환: list of dict — process_single_video 시그니처 참고.
    """
    if not ACTIVE_DETECTORS:
        raise RuntimeError("ACTIVE_DETECTORS 가 비어있음 — 최소 하나의 detector 키가 필요합니다.")
    missing = [k for k in ACTIVE_DETECTORS if k not in DETECTORS]
    if missing:
        raise KeyError(f"ACTIVE_DETECTORS 의 키가 DETECTORS 레지스트리에 없음: {missing}")

    specs = []
    for key in ACTIVE_DETECTORS:
        cfg = DETECTORS[key]
        path          = cfg["path"]
        info_path     = cfg["info"]
        allowed_names = cfg.get("allowed_names")

        print(f"  [{key}] YOLO 로딩: {path}")
        model, resolved_path = _load_yolo_detection_model(path)
        print(f"    -> {resolved_path}")

        full_label_mapping = load_class_info(info_path)
        print(f"    info YAML: {info_path} ({len(full_label_mapping)} classes)")

        allowed_ids, filtered_mapping = derive_allowed_class_ids(full_label_mapping, allowed_names)
        if allowed_ids is None:
            label_mapping   = full_label_mapping
            allowed_classes = None
        else:
            label_mapping   = filtered_mapping
            allowed_classes = allowed_ids
            print(f"    허용 클래스 id={sorted(allowed_ids)} names={sorted(filtered_mapping.values())}")

        # 패널 부제: display_name 우선, 없으면 모델 파일 stem 폴백
        title = cfg.get("display_name") or resolved_path.stem
        # 출력 파일명용 tag 는 항상 모델 stem 기반 (display_name 의 공백/특수문자 회피)
        tag = _sanitize_filename_component(resolved_path.stem, fallback=key)
        stage2 = bool(cfg.get("stage2", False))
        print(f"    stage2(이 패널에 분류기 적용): {stage2}")

        specs.append({
            "name"           : key,
            "title"          : title,
            "tag"            : tag,
            "model"          : model,
            "label_mapping"  : label_mapping,
            "allowed_classes": allowed_classes,
            "stage2"         : stage2,
        })

    return specs


# ───────────────────────────── Worker 측 (자식 프로세스) ─────────────────────────────
# 'spawn' 으로 띄우는 자식 프로세스에서:
#   1) _worker_init() 가 한 번 실행되어 모델(들)/분류기를 로드 → 모듈 전역 _WORKER_STATE 보관
#   2) 이후 _worker_process(video_path, batch_timestamp) 가 영상마다 호출되어 재사용
# YOLO/Classifier 객체는 picklable 이 아니므로 부모↔자식 사이로 직접 보내지 않고
# 자식 프로세스에서 직접 로드한다.
_WORKER_STATE: dict = {}


def _progress_iter(iterable, total, desc):
    """tqdm 호환 iterator.

    - 직렬 모드(부모 프로세스): 평소처럼 tqdm 진행바를 그대로 표시.
    - 병렬 모드(자식 워커): tqdm 을 직접 그리는 대신 multiprocessing Queue 로
      (worker_id, 영상명, 현재 frame, 총 frame) 진행상황을 부모에 전송.
      부모 프로세스는 슬롯별 tqdm bar 를 한 화면에 모아 실시간으로 갱신한다.
    """
    pq = _WORKER_STATE.get("progress_queue")
    wid = _WORKER_STATE.get("worker_id")
    if pq is None or wid is None:
        # 직렬 경로: 평소처럼 tqdm
        return tqdm(iterable, total=total, desc=f"  {desc}", mininterval=0.5)

    # 병렬 경로: Queue 기반 progress 이벤트 generator
    def _gen():
        try:
            pq.put(("start", wid, desc, total))
        except Exception:
            pass
        # 너무 잦은 put 은 IPC 비용이 부담 → 최대 ~200 회로 throttle
        step = max(1, total // 200) if total else 1
        last_emit = 0
        i = 0
        for item in iterable:
            i += 1
            yield item
            if (i - last_emit) >= step or i == total:
                try:
                    pq.put(("progress", wid, i))
                except Exception:
                    pass
                last_emit = i
        try:
            pq.put(("done", wid))
        except Exception:
            pass
    return _gen()


def _worker_init(threads_per_worker: int,
                 progress_queue=None,
                 worker_counter=None):
    """ProcessPoolExecutor 워커 초기화: 스레드 제한 + 모델 로드 + worker_id 부여."""
    # ── intra-op 스레드 제한 (oversubscription 방지) ──
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(threads_per_worker)

    try:
        import torch
        torch.set_num_threads(threads_per_worker)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception as e:
        print(f"  [worker init] torch thread 설정 실패: {e}")

    try:
        cv2.setNumThreads(threads_per_worker)
    except Exception:
        pass

    # ── worker slot id (0..N-1) 할당: 부모가 넘긴 공유 Counter 를 원자적으로 증가 ──
    if progress_queue is not None and worker_counter is not None:
        with worker_counter.get_lock():
            wid = int(worker_counter.value)
            worker_counter.value = wid + 1
        _WORKER_STATE["worker_id"]      = wid
        _WORKER_STATE["progress_queue"] = progress_queue

    # ── 모델/분류기 로드 (1회만) ──
    engines_spec = _build_engines_spec()
    tracker_factory, device = _make_tracker_factory()

    classifier_bundle = None
    need_classifier = (
        ACTIVE_CLASSIFIER is not None
        and any(DETECTORS[k].get("stage2", False) for k in ACTIVE_DETECTORS)
    )
    if need_classifier:
        classifier_bundle = load_classifier_bundle(ACTIVE_CLASSIFIER, device)

    _WORKER_STATE["engines_spec"]      = engines_spec
    _WORKER_STATE["tracker_factory"]   = tracker_factory
    _WORKER_STATE["classifier_bundle"] = classifier_bundle
    _WORKER_STATE["device"]            = device
    _WORKER_STATE["threads"]           = threads_per_worker


def _worker_process_one(video_path_str: str, batch_timestamp: str):
    """워커에서 영상 1개를 처리하고 (결과, 로그) 를 반환.

    워커의 stdout/stderr 를 메모리 버퍼로 잡아 부모 프로세스로 돌려보내,
    부모가 영상 단위로 깔끔히 출력할 수 있게 한다. 진행상황은 _progress_iter 가
    Queue 로 부모에 별도 전송하므로 화면에는 라이브 bar 로 표시된다.
    """
    import io
    import contextlib
    import traceback

    vp = Path(video_path_str)
    buf = io.StringIO()
    res = None
    err = None
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            cfg_path = Path(CONFIG_DIR) / f"{vp.stem}.json"
            cfg = load_video_config(cfg_path)
            if cfg is None:
                print(f"    [skip] config not found: {cfg_path}")
            else:
                res = process_single_video(
                    vp, cfg,
                    _WORKER_STATE["engines_spec"],
                    _WORKER_STATE["tracker_factory"],
                    batch_timestamp,
                    classifier_bundle=_WORKER_STATE["classifier_bundle"],
                )
        except Exception:
            err = traceback.format_exc()
            print(err)
    return {
        "video": vp.name,
        "result": res,
        "log": buf.getvalue(),
        "error": err,
    }


# ───────────────────────────── Parent 측 ─────────────────────────────

def _resolve_parallelism(num_videos: int):
    """MAX_PARALLEL_VIDEOS / THREADS_PER_WORKER 의 'auto' 를 정수로 해결.

    물리 코어 수는 hyper-threading 을 가정하고 logical CPU / 2 로 추정.
    """
    try:
        logical = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        logical = os.cpu_count() or 1
    physical = max(1, logical // 2)

    if isinstance(MAX_PARALLEL_VIDEOS, str) and MAX_PARALLEL_VIDEOS.lower() == "auto":
        hint = max(1, int(THREADS_PER_WORKER_HINT))
        workers = max(1, min(num_videos, max(1, physical // hint)))
    else:
        workers = max(1, min(int(MAX_PARALLEL_VIDEOS), num_videos))

    if isinstance(THREADS_PER_WORKER, str) and THREADS_PER_WORKER.lower() == "auto":
        threads = max(1, physical // workers)
    else:
        threads = max(1, int(THREADS_PER_WORKER))

    return workers, threads, physical, logical


def _processing_phase_sequential(video_paths, batch_timestamp):
    """병렬 비활성(워커=1) 시 사용하는 기존 직렬 경로."""
    engines_spec = _build_engines_spec()

    print("  BoTSORT 트래커 팩토리 초기화...")
    tracker_factory, device = _make_tracker_factory()
    print(f"  디바이스: {device}")

    classifier_bundle = None
    need_classifier = (
        ACTIVE_CLASSIFIER is not None
        and any(DETECTORS[k].get("stage2", False) for k in ACTIVE_DETECTORS)
    )
    if need_classifier:
        classifier_bundle = load_classifier_bundle(ACTIVE_CLASSIFIER, device)
    elif ACTIVE_CLASSIFIER is not None:
        print("  [info] ACTIVE_CLASSIFIER 가 지정됐지만 DETECTORS[*]['stage2']=True 인 패널이 없어 Stage-2 를 건너뜀")

    results = []
    for vi, vp in enumerate(video_paths, 1):
        cfg_path = Path(CONFIG_DIR) / f"{vp.stem}.json"
        cfg = load_video_config(cfg_path)
        print(f"\n  [{vi}/{len(video_paths)}] {vp.name}")
        if cfg is None:
            print(f"    [skip] config not found: {cfg_path}")
            continue
        res = process_single_video(
            vp, cfg, engines_spec, tracker_factory, batch_timestamp,
            classifier_bundle=classifier_bundle,
        )
        if res is not None:
            results.append(res)
    return results


def _processing_phase_parallel(video_paths, batch_timestamp, workers, threads):
    """ProcessPoolExecutor 기반 영상-단위 병렬 처리.

    실시간 UX:
      - 워커가 보내는 progress 메시지를 부모의 디스플레이 스레드가 받아
        워커 슬롯별 tqdm bar (position=0..workers-1) 로 화면 하단에 고정 표시.
      - 영상이 끝날 때마다 부모는 `tqdm.write()` 로 진행바 위에 캡처된 로그를
        깔끔하게 출력 (bar 위치를 깨뜨리지 않음).
      - 마지막 줄은 전체 영상 진행률(`Overall`).
    """
    import multiprocessing as mp
    import threading
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ctx = mp.get_context("spawn")
    n = len(video_paths)
    print(f"  영상 단위 병렬 처리: workers={workers}, threads/worker={threads}")
    print(f"  (각 워커가 자체적으로 모델을 로드합니다 — 초기 로딩에 수십 초가 걸릴 수 있음)")

    # 큰 영상이 먼저 끝나도록 파일 크기 내림차순으로 제출 (straggler 완화)
    try:
        ordered = sorted(video_paths, key=lambda p: -Path(p).stat().st_size)
    except Exception:
        ordered = list(video_paths)
    idx_of = {vp: i + 1 for i, vp in enumerate(video_paths)}

    # ── IPC 자원: 진행상황 큐 + worker_id 카운터 ──
    progress_queue = ctx.Queue()
    worker_counter = ctx.Value("i", 0)  # 내장 lock 보유

    # ── 진행바 디스플레이 스레드 (부모) ──
    # position 0..workers-1 = 워커별 라이브 bar
    # position workers      = 전체 영상 진행률
    overall = tqdm(total=n, position=workers, desc="Overall",
                   unit="vid", leave=True, dynamic_ncols=True)
    bars: dict = {}                  # wid -> tqdm bar
    cur_video: dict = {}             # wid -> 현재 처리중인 영상 stem
    stop_event = threading.Event()

    def _display_loop():
        while not stop_event.is_set():
            try:
                msg = progress_queue.get(timeout=0.3)
            except Exception:
                continue
            if msg is None:
                break
            kind = msg[0]
            if kind == "start":
                _, wid, desc, total = msg
                cur_video[wid] = desc
                if wid in bars:
                    bars[wid].reset(total=total)
                    bars[wid].set_description_str(f"[w{wid}] {desc}")
                else:
                    bars[wid] = tqdm(
                        total=total,
                        position=wid,
                        desc=f"[w{wid}] {desc}",
                        unit="f",
                        leave=False,
                        dynamic_ncols=True,
                        mininterval=0.3,
                    )
            elif kind == "progress":
                _, wid, current = msg
                bar = bars.get(wid)
                if bar is not None:
                    bar.n = current
                    bar.refresh()
            elif kind == "done":
                _, wid = msg
                bar = bars.get(wid)
                if bar is not None:
                    if bar.total:
                        bar.n = bar.total
                    bar.refresh()
                cur_video.pop(wid, None)
            elif kind == "log":
                # 메인 출력 영역에 영상별 캡처 로그 dump (bars 위쪽으로 흘러나감)
                _, text = msg
                tqdm.write(text)
            elif kind == "video_done":
                overall.update(1)

    display_thread = threading.Thread(target=_display_loop, daemon=True)
    display_thread.start()

    results = []
    t0 = time.perf_counter()
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(threads, progress_queue, worker_counter),
        ) as ex:
            future_to_vp = {
                ex.submit(_worker_process_one, str(vp), batch_timestamp): vp
                for vp in ordered
            }
            # 제출 목록은 bar 시작 전에 한 번만 출력
            for vp in ordered:
                tqdm.write(f"  [submit {idx_of[vp]:>2}/{n}] {Path(vp).name}")

            completed = 0
            for fut in as_completed(future_to_vp):
                vp = future_to_vp[fut]
                completed += 1
                try:
                    payload = fut.result()
                except Exception as e:
                    progress_queue.put(("log", f"\n  [worker fatal] {vp.name}: {e!r}"))
                    progress_queue.put(("video_done",))
                    continue
                elapsed_so_far = time.perf_counter() - t0
                header = (
                    "\n" + "┌" + "─" * 72 + "\n"
                    f"│ [done {completed:>2}/{n}] {payload['video']}   "
                    f"(wall={elapsed_so_far:.1f}s)\n"
                    "└" + "─" * 72
                )
                progress_queue.put(("log", header))
                log = payload.get("log", "")
                if log:
                    progress_queue.put(("log", log.rstrip()))
                progress_queue.put(("video_done",))
                res = payload.get("result")
                if res is not None:
                    results.append(res)
    finally:
        # 디스플레이 스레드 종료
        stop_event.set()
        try:
            progress_queue.put(None)
        except Exception:
            pass
        display_thread.join(timeout=2)
        for b in list(bars.values()):
            try:
                b.close()
            except Exception:
                pass
        try:
            overall.close()
        except Exception:
            pass
    return results


def processing_phase(video_paths, batch_timestamp):
    """영상 배치 처리 진입점.

    MAX_PARALLEL_VIDEOS 가 1 이거나 영상이 1개뿐이면 직렬 경로,
    그 외에는 영상-단위 멀티프로세스 경로로 분기한다.
    """
    print()
    print("=" * 60)
    print(f"  Processing phase ({len(video_paths)} videos, batch={batch_timestamp})")
    compare_flag = len(ACTIVE_DETECTORS) >= 2
    print(f"  ACTIVE_DETECTORS={ACTIVE_DETECTORS}  compare={compare_flag}  COUNTING_MODE={COUNTING_MODE}")
    print(f"  ACTIVE_CLASSIFIER={ACTIVE_CLASSIFIER}")

    workers, threads, physical, logical = _resolve_parallelism(len(video_paths))
    print(f"  CPU: {logical} logical / {physical} physical (est.)  → "
          f"workers={workers}, threads/worker={threads}")
    print("=" * 60)

    t_phase_start = time.perf_counter()
    if workers <= 1 or len(video_paths) <= 1:
        results = _processing_phase_sequential(video_paths, batch_timestamp)
    else:
        results = _processing_phase_parallel(video_paths, batch_timestamp, workers, threads)
    phase_elapsed = time.perf_counter() - t_phase_start

    # ── 전체 요약 ──
    if results:
        print()
        print("=" * 60)
        print("  배치 전체 요약")
        print("=" * 60)
        grand_total = 0
        grand_frames = 0
        grand_elapsed = 0.0
        for r in results:
            grand_total += r["total"]
            grand_frames += r["frames"]
            grand_elapsed += r["elapsed"]
            print(f"  {r['video']:<40} frames={r['frames']:>6} "
                  f"total_count={r['total']:>6} elapsed={r['elapsed']:>6.1f}s")
        print(f"  {'─' * 40}")
        print(f"  Videos: {len(results)}  Frames: {grand_frames}  "
              f"Total count: {grand_total}  Elapsed(sum of per-video): {grand_elapsed:.1f}s")
        print(f"  Wall-clock (processing_phase): {phase_elapsed:.1f}s  "
              f"→ speedup ≈ {grand_elapsed / max(phase_elapsed, 1e-6):.2f}×")
        print("=" * 60)


# ═══════════════════════════════════════════════════════════════
# 메인 실행
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  교통량 카운팅 시각화 영상 생성")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)
    DETECTION_DIR.mkdir(parents=True, exist_ok=True)
    CLASSIFICATION_DIR.mkdir(parents=True, exist_ok=True)

    video_paths = collect_video_paths()
    if not video_paths:
        return

    print(f"  대상 영상 {len(video_paths)}개:")
    for vp in video_paths:
        print(f"    - {vp}")
    print(f"  RUN_MODE={RUN_MODE}  ROI_MODE={ROI_MODE}  OVERWRITE_EXISTING_CONFIG={OVERWRITE_EXISTING_CONFIG}")
    print(f"  COUNTING_MODE={COUNTING_MODE}  COLOR_MODE={COLOR_MODE}")
    print(f"  USE_TRACKER_BBOX={USE_TRACKER_BBOX}  SHOW_TRAJECTORY={SHOW_TRAJECTORY}  HIDDEN_CLASSES={sorted(HIDDEN_CLASSES)}")
    print(f"  ACTIVE_DETECTORS={ACTIVE_DETECTORS}  (compare={'ON' if len(ACTIVE_DETECTORS) >= 2 else 'OFF'})")
    print(f"  ACTIVE_CLASSIFIER={ACTIVE_CLASSIFIER}  DETECTION_DIR={DETECTION_DIR}")
    for k in ACTIVE_DETECTORS:
        cfg = DETECTORS.get(k, {})
        print(f"    - {k}: path={cfg.get('path')}  info={cfg.get('info')}  "
              f"allowed_names={cfg.get('allowed_names')}  stage2={cfg.get('stage2', False)}")

    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    headless = _detect_headless()
    effective_run_mode = RUN_MODE
    if headless and RUN_MODE in ("draw_only", "draw_and_process"):
        if RUN_MODE == "draw_only":
            print()
            print("  [headless] GUI 가 없는 환경에서 RUN_MODE='draw_only' 는 실행 불가합니다.")
            print("  [headless] GUI 가 있는 PC 에서 먼저 그리기 페이즈를 수행한 뒤,")
            print("             configs/*.json 과 input/ 영상을 이 서버로 복사해 RUN_MODE='process_only' 로 다시 실행하세요.")
            return
        # draw_and_process → process_only 로 다운그레이드
        print()
        print("  [headless] GUI 미감지 → drawing_phase 를 건너뛰고 RUN_MODE='process_only' 로 강등합니다.")
        print("             (configs/*.json 이 이미 준비돼 있어야 합니다)")
        effective_run_mode = "process_only"

    print(f"  HEADLESS(detected)={headless}  effective RUN_MODE={effective_run_mode}")

    if effective_run_mode in ("draw_only", "draw_and_process"):
        drawing_phase(video_paths)

    if effective_run_mode in ("process_only", "draw_and_process"):
        processing_phase(video_paths, batch_timestamp)


if __name__ == "__main__":
    main()
