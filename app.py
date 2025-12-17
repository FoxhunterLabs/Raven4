# ================================================================
# RAVEN 4 · Rivian Autonomy & Visual Energy Nexus
# ------------------------------------------------
# “Brain-on-a-table” deterministic autonomy cockpit demo:
# - Synthetic 3D trail (x,y,z) with grade/roughness/edge/traction
# - Drift + stability scoring (pure math)
# - Deterministic modes with hysteresis + human-gated safety
# - NDJSON audit log + one-click download
# - 3D OrbitView (deck.gl / PyDeck) + HUD + event timeline
#
# No ML. No cloud. No black box.
# ================================================================

import math
import time
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# ----------------------------
# Page + Theme
# ----------------------------
st.set_page_config(page_title="RAVEN 4 · Autonomy Kernel Cockpit", layout="wide")

BG = "#020714"
PANEL = "#060b1a"
ACCENT = "#00e0a4"
WARN = "#ffb454"
DANGER = "#ff4d4d"
MUTED = "#9ca3af"

st.markdown(
    f"""
<style>
.stApp {{
  background: radial-gradient(circle at top, #06141a 0, {BG} 45%, #000 100%);
  color: white;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
}}

.block-container {{ padding-top: 1.1rem; }}

.r4-title {{
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: {ACCENT};
  font-weight: 700;
  font-size: 1.15rem;
}}

.r4-sub {{
  color: {MUTED};
  font-size: 0.92rem;
  margin-top: 0.25rem;
}}

.r4-card {{
  background: linear-gradient(180deg, rgba(10,14,32,0.86), rgba(3,6,16,0.92));
  border: 1px solid rgba(148,163,184,0.12);
  border-radius: 18px;
  padding: 0.85rem 0.95rem;
}}

.r4-chip {{
  display:inline-block;
  padding: 0.18rem 0.55rem;
  border-radius: 999px;
  font-size: 0.72rem;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  border: 1px solid rgba(148,163,184,0.16);
  background: rgba(15,23,42,0.50);
  color: {MUTED};
}}

.r4-chip-ok {{
  border-color: rgba(0,224,164,0.35);
  color: rgba(0,224,164,0.95);
}}

.r4-chip-warn {{
  border-color: rgba(255,180,84,0.35);
  color: rgba(255,180,84,0.95);
}}

.r4-chip-danger {{
  border-color: rgba(255,77,77,0.38);
  color: rgba(255,77,77,0.95);
}}

.r4-metric-label {{
  color: {MUTED};
  font-size: 0.72rem;
  letter-spacing: 0.10em;
  text-transform: uppercase;
}}

.r4-metric-value {{
  font-size: 1.30rem;
  font-weight: 750;
  margin-top: -0.05rem;
}}

hr {{
  border: none;
  border-top: 1px solid rgba(148,163,184,0.12);
  margin: 0.75rem 0;
}}

.small-muted {{ color: {MUTED}; font-size: 0.85rem; }}

</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Models
# ----------------------------
@dataclass
class TerrainSegment:
    i: int
    x: float
    y: float
    z: float
    slope_deg: float
    roughness: float        # 0–1
    edge_exposure: float    # 0–1
    traction_coeff: float   # 0–1
    curvature: float        # signed (turn intensity)
    risk: float             # 0–1 composite for viz


@dataclass
class VehicleState:
    x: float
    y: float
    z: float
    speed_mps: float
    stability_index: float  # 0–100 (higher better)
    drift_score: float      # 0–100 (higher worse)
    grade_deg: float
    mode: str               # HOLD, CRUISE, CAUTIOUS, CRAWL, STOP_SAFE


@dataclass
class DecisionEvent:
    tick: int
    seg_i: int
    action: str
    reason: str
    grade_deg: float
    drift_score: float
    stability_index: float
    human_required: bool
    human_override: str     # PENDING, APPROVED, DENIED, AUTO
    timestamp: float
    thresholds: Dict[str, float]


# ----------------------------
# Audit logging
# ----------------------------
RUNS_DIR = Path("runs_raven4")
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _now_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


def current_run_id() -> str:
    if "r4_run_id" not in st.session_state:
        st.session_state.r4_run_id = _now_run_id()
    return st.session_state.r4_run_id


def audit_append(record: Dict) -> None:
    fpath = RUNS_DIR / f"{current_run_id()}.ndjson"
    with fpath.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def audit_bytes_for_download() -> bytes:
    fpath = RUNS_DIR / f"{current_run_id()}.ndjson"
    if not fpath.exists():
        return b""
    return fpath.read_bytes()


# ----------------------------
# Terrain generation (3D, consistent grade)
# ----------------------------
def _preset_seed(preset: str) -> int:
    return {"Mountain ridge": 42, "Forest trail": 17, "Desert wash": 9}.get(preset, 42)


def generate_terrain(preset: str, n: int = 180, step: float = 1.0) -> List[TerrainSegment]:
    """
    Synthetic 3D trail:
    - x,y: plan-view path with curvature
    - z: elevation integrated from slope
    """
    rng = np.random.default_rng(_preset_seed(preset))

    # Envelopes
    if preset == "Mountain ridge":
        max_grade = 18.0
        rough_base = (0.28, 0.72)
        edge_base = (0.35, 0.85)
        traction_base = (0.45, 0.75)
        curv_freq = 7.0
        slope_freq = 2.3
    elif preset == "Forest trail":
        max_grade = 12.0
        rough_base = (0.18, 0.55)
        edge_base = (0.10, 0.45)
        traction_base = (0.60, 0.92)
        curv_freq = 4.0
        slope_freq = 1.4
    else:  # Desert wash
        max_grade = 9.0
        rough_base = (0.30, 0.85)
        edge_base = (0.18, 0.60)
        traction_base = (0.35, 0.70)
        curv_freq = 5.5
        slope_freq = 1.0

    x, y, z = 0.0, 0.0, 0.0
    heading = 0.0

    segments: List[TerrainSegment] = []
    for i in range(n):
        t = i / max(1, n - 1)

        curvature = math.sin(t * curv_freq * math.pi) * (0.5 + 0.3 * rng.random())
        heading += curvature * 0.08

        slope_pattern = math.sin(t * slope_freq * math.pi) + 0.25 * math.sin(t * (slope_freq * 2.1) * math.pi)
        slope_deg = float(np.clip(slope_pattern * max_grade, -max_grade, max_grade))
        slope_rad = math.radians(slope_deg)

        rough = float(rng.uniform(*rough_base))
        edge = float(rng.uniform(*edge_base) * (0.75 + 0.40 * abs(curvature)))
        edge = float(np.clip(edge, 0.0, 1.0))
        traction = float(rng.uniform(*traction_base))

        # integrate geometry
        x += math.cos(heading) * step
        y += math.sin(heading) * step
        z += math.tan(slope_rad) * step

        # composite risk just for viz
        risk = 0.36 * rough + 0.40 * edge + 0.24 * min(1.0, abs(slope_deg) / 18.0)
        risk = float(np.clip(risk, 0.0, 1.0))

        segments.append(
            TerrainSegment(
                i=i,
                x=x,
                y=y,
                z=z,
                slope_deg=slope_deg,
                roughness=rough,
                edge_exposure=edge,
                traction_coeff=traction,
                curvature=float(curvature),
                risk=risk,
            )
        )

    return segments


# ----------------------------
# Deterministic scoring + decisions
# ----------------------------
def evaluate_segment(seg: TerrainSegment, vehicle: VehicleState, weather_factor: float) -> Dict[str, float]:
    """
    Math-only scoring:
    - grade_risk around 8–22° envelope
    - drift from grade, traction, roughness, edge, speed
    - stability from drift + weather
    """
    grade = seg.slope_deg
    grade_risk = max(0.0, (abs(grade) - 8.0) / 14.0)  # 0..~1
    speed_factor = min(1.0, vehicle.speed_mps / 16.0)

    traction_risk = (1.0 - seg.traction_coeff) * (0.40 + 0.60 * speed_factor)
    rough_risk = seg.roughness * (0.30 + 0.70 * speed_factor)
    edge_risk = seg.edge_exposure * (0.35 + 0.65 * speed_factor)

    drift_risk = 0.35 * grade_risk + 0.30 * traction_risk + 0.25 * rough_risk + 0.10 * edge_risk

    instab_raw = 0.42 * drift_risk + 0.30 * rough_risk + 0.18 * grade_risk + 0.10 * weather_factor

    drift_score = float(np.clip(drift_risk * 100.0, 0.0, 100.0))
    stability_index = float(np.clip(100.0 - instab_raw * 100.0, 0.0, 100.0))

    return {"grade_deg": float(grade), "drift_score": drift_score, "stability_index": stability_index}


def _adjust_threshold(base: float, risk_appetite: float, span: float) -> float:
    """risk_appetite: 0 conservative → base-span; 1 aggressive → base+span"""
    return base + (risk_appetite - 0.5) * 2.0 * span


def choose_action(
    metrics: Dict[str, float],
    vehicle: VehicleState,
    clarity_floor: float,
    risk_appetite: float,
) -> Tuple[Dict[str, object], Dict[str, float]]:
    """
    Deterministic decision layer with:
    - thresholds adjusted by risk appetite (envelope tuning)
    - hysteresis to prevent thrash
    - human-gating for STOP_SAFE / CRAWL + low stability (clarity_floor)
    """
    drift = metrics["drift_score"]
    stab = metrics["stability_index"]
    grade = metrics["grade_deg"]

    prev_mode = vehicle.mode if vehicle.mode and vehicle.mode != "HOLD" else "CRUISE"

    base = {
        "drift": {"CAUTIOUS": 40.0, "CRAWL": 60.0, "STOP_SAFE": 80.0},
        "stab": {"CAUTIOUS": 65.0, "CRAWL": 50.0, "STOP_SAFE": 35.0},
        "grade": {"CAUTIOUS": 10.0, "CRAWL": 14.0, "STOP_SAFE": 18.0},
    }

    drift_th = {k: _adjust_threshold(v, risk_appetite, span=8.0) for k, v in base["drift"].items()}
    # conservative drivers require MORE stability, so invert appetite for stab thresholds
    inv = 1.0 - risk_appetite
    stab_th = {
        "CAUTIOUS": _adjust_threshold(base["stab"]["CAUTIOUS"], inv, span=8.0),
        "CRAWL": _adjust_threshold(base["stab"]["CRAWL"], inv, span=8.0),
        "STOP_SAFE": _adjust_threshold(base["stab"]["STOP_SAFE"], inv, span=8.0),
    }
    grade_th = {k: _adjust_threshold(v, risk_appetite, span=3.0) for k, v in base["grade"].items()}

    thresholds_flat = {
        "drift_CAUTIOUS": drift_th["CAUTIOUS"],
        "drift_CRAWL": drift_th["CRAWL"],
        "drift_STOP_SAFE": drift_th["STOP_SAFE"],
        "stab_CAUTIOUS": stab_th["CAUTIOUS"],
        "stab_CRAWL": stab_th["CRAWL"],
        "stab_STOP_SAFE": stab_th["STOP_SAFE"],
        "grade_CAUTIOUS": grade_th["CAUTIOUS"],
        "grade_CRAWL": grade_th["CRAWL"],
        "grade_STOP_SAFE": grade_th["STOP_SAFE"],
        "clarity_floor": float(clarity_floor),
    }

    reason_chunks = []
    human_required = False

    # Raw mode
    if drift > drift_th["STOP_SAFE"] or stab < stab_th["STOP_SAFE"] or abs(grade) > grade_th["STOP_SAFE"]:
        raw = "STOP_SAFE"
        reason_chunks.append("beyond hard safety envelope")
    elif drift > drift_th["CRAWL"] or stab < stab_th["CRAWL"] or abs(grade) > grade_th["CRAWL"]:
        raw = "CRAWL"
        reason_chunks.append("high risk region (crawl envelope)")
    elif drift > drift_th["CAUTIOUS"] or stab < stab_th["CAUTIOUS"] or abs(grade) > grade_th["CAUTIOUS"]:
        raw = "CAUTIOUS"
        reason_chunks.append("moderate risk region (cautious envelope)")
    else:
        raw = "CRUISE"
        reason_chunks.append("within stable window")

    # Hysteresis (de-escalation requires extra margin)
    H_DRIFT = 5.0
    H_STAB = 5.0
    action = raw

    if prev_mode == "STOP_SAFE" and raw in ("CRAWL", "CAUTIOUS", "CRUISE"):
        if not (drift < drift_th["STOP_SAFE"] - H_DRIFT and stab > stab_th["STOP_SAFE"] + H_STAB):
            action = "STOP_SAFE"
            reason_chunks.append("hysteresis hold: STOP_SAFE")
    elif prev_mode == "CRAWL" and raw in ("CAUTIOUS", "CRUISE"):
        if not (drift < drift_th["CRAWL"] - H_DRIFT and stab > stab_th["CRAWL"] + H_STAB):
            action = "CRAWL"
            reason_chunks.append("hysteresis hold: CRAWL")
    elif prev_mode == "CAUTIOUS" and raw == "CRUISE":
        if not (drift < drift_th["CAUTIOUS"] - H_DRIFT and stab > stab_th["CAUTIOUS"] + H_STAB):
            action = "CAUTIOUS"
            reason_chunks.append("hysteresis hold: CAUTIOUS")

    # Human attention
    if stab < clarity_floor:
        human_required = True
        reason_chunks.append("stability below attention floor")

    # Always human-gate the most critical modes
    if action in ("STOP_SAFE", "CRAWL"):
        human_required = True

    # Speed targets
    target_speed = {"STOP_SAFE": 0.0, "CRAWL": 2.0, "CAUTIOUS": 6.0, "CRUISE": 12.0}[action]
    alpha = 0.25
    new_speed = vehicle.speed_mps + alpha * (target_speed - vehicle.speed_mps)

    return (
        {
            "action": action,
            "reason": ", ".join(reason_chunks),
            "speed_mps": float(new_speed),
            "human_required": bool(human_required),
        },
        thresholds_flat,
    )


# ----------------------------
# Session state
# ----------------------------
def init_state(preset: str = "Mountain ridge") -> None:
    terrain = generate_terrain(preset)
    v0 = VehicleState(
        x=terrain[0].x,
        y=terrain[0].y,
        z=terrain[0].z,
        speed_mps=0.0,
        stability_index=100.0,
        drift_score=0.0,
        grade_deg=0.0,
        mode="HOLD",
    )
    st.session_state.r4 = {
        "tick": 0,
        "terrain": terrain,
        "vehicle": v0,
        "path": [(v0.x, v0.y, v0.z)],
        "decisions": [],  # List[DecisionEvent]
        "human_lock_active": False,
        "human_lock_index": None,
        "paused": True,
        "autoplay_hz": 6.0,
        "preset": preset,
        "scrub_tick": 0,
    }
    st.session_state.r4_run_id = _now_run_id()


if "r4" not in st.session_state:
    init_state("Mountain ridge")


def reset_run(preset: str) -> None:
    init_state(preset)


def step_raven(params: Dict[str, float]) -> None:
    s = st.session_state.r4
    s["tick"] += 1
    tick = s["tick"]

    terrain: List[TerrainSegment] = s["terrain"]
    v: VehicleState = s["vehicle"]

    seg_i = min(len(terrain) - 1, tick)
    seg = terrain[seg_i]

    metrics = evaluate_segment(seg, v, weather_factor=params["weather_factor"])
    decision, thresholds = choose_action(
        metrics=metrics,
        vehicle=v,
        clarity_floor=params["clarity_floor"],
        risk_appetite=params["risk_appetite"],
    )

    # Update vehicle
    v.x, v.y, v.z = seg.x, seg.y, seg.z
    v.grade_deg = metrics["grade_deg"]
    v.drift_score = metrics["drift_score"]
    v.stability_index = metrics["stability_index"]
    v.speed_mps = decision["speed_mps"]
    v.mode = decision["action"]
    s["vehicle"] = v
    s["path"].append((v.x, v.y, v.z))

    ev = DecisionEvent(
        tick=tick,
        seg_i=seg_i,
        action=decision["action"],
        reason=decision["reason"],
        grade_deg=v.grade_deg,
        drift_score=v.drift_score,
        stability_index=v.stability_index,
        human_required=decision["human_required"],
        human_override="PENDING" if decision["human_required"] else "AUTO",
        timestamp=time.time(),
        thresholds=thresholds,
    )
    s["decisions"].append(ev)

    if ev.human_required and not s["human_lock_active"]:
        s["human_lock_active"] = True
        s["human_lock_index"] = len(s["decisions"]) - 1

    audit_append(
        {
            "run_id": current_run_id(),
            "tick": tick,
            "params": params,
            "terrain": asdict(seg),
            "vehicle": asdict(v),
            "metrics": metrics,
            "decision": {
                "action": ev.action,
                "reason": ev.reason,
                "human_required": ev.human_required,
                "human_override": ev.human_override,
                "thresholds": thresholds,
            },
            "ts": ev.timestamp,
        }
    )

    st.session_state.r4 = s


# ----------------------------
# 3D Viz (PyDeck OrbitView)
# ----------------------------
def _risk_to_rgb(r: float) -> List[int]:
    """
    0..1 risk -> green->amber->red.
    """
    r = float(np.clip(r, 0.0, 1.0))
    if r < 0.55:
        # green-ish
        return [int(20 + 40 * r), int(200 + 35 * (1 - r)), int(150)]
    if r < 0.82:
        # amber
        t = (r - 0.55) / (0.27)
        return [int(255), int(170 + 50 * (1 - t)), int(70)]
    # red
    t = (r - 0.82) / (0.18)
    return [int(255), int(90 * (1 - t)), int(90 * (1 - t))]


def build_deck_scene(
    terrain: List[TerrainSegment],
    vehicle: VehicleState,
    path: List[Tuple[float, float, float]],
    focus_tick: Optional[int] = None,
) -> pdk.Deck:
    # segments for LineLayer: each row needs source + target
    seg_rows = []
    for i in range(1, len(terrain)):
        a = terrain[i - 1]
        b = terrain[i]
        seg_rows.append(
            {
                "src": [a.x, a.y, a.z],
                "dst": [b.x, b.y, b.z],
                "color": _risk_to_rgb(b.risk),
                "i": i,
                "risk": b.risk,
                "grade": b.slope_deg,
            }
        )

    # driven path polyline as thicker "neon" line
    path_rows = []
    if len(path) >= 2:
        for i in range(1, len(path)):
            ax, ay, az = path[i - 1]
            bx, by, bz = path[i]
            path_rows.append({"src": [ax, ay, az], "dst": [bx, by, bz]})

    # focus highlight (scrub to any tick)
    highlight_rows = []
    if focus_tick is not None and 1 <= focus_tick < len(terrain):
        a = terrain[max(0, focus_tick - 1)]
        b = terrain[focus_tick]
        highlight_rows.append({"src": [a.x, a.y, a.z], "dst": [b.x, b.y, b.z]})

    trail_layer = pdk.Layer(
        "LineLayer",
        data=seg_rows,
        get_source_position="src",
        get_target_position="dst",
        get_color="color",
        get_width=2.6,
        width_units="pixels",
        pickable=True,
        auto_highlight=True,
    )

    driven_layer = pdk.Layer(
        "LineLayer",
        data=path_rows,
        get_source_position="src",
        get_target_position="dst",
        get_color=[0, 224, 164],
        get_width=4.4,
        width_units="pixels",
        pickable=False,
    )

    highlight_layer = pdk.Layer(
        "LineLayer",
        data=highlight_rows,
        get_source_position="src",
        get_target_position="dst",
        get_color=[255, 180, 84],
        get_width=7.0,
        width_units="pixels",
        pickable=False,
    )

    vehicle_layer = pdk.Layer(
        "ScatterplotLayer",
        data=[{"pos": [vehicle.x, vehicle.y, vehicle.z], "speed": vehicle.speed_mps}],
        get_position="pos",
        get_fill_color=[0, 224, 164],
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
        get_radius=3.6,
        radius_units="meters",
        pickable=True,
    )

    # Camera target near the vehicle (OrbitView is in cartesian)
    view_state = pdk.ViewState(
        target=[vehicle.x, vehicle.y, vehicle.z],
        rotationOrbit=18,
        rotationX=54,
        zoom=0.9,
        minZoom=-2,
        maxZoom=5,
    )

    tooltip = {
        "html": """
        <div style="font-family: ui-sans-serif; font-size: 12px;">
          <div><b>Segment</b> {i}</div>
          <div>Risk: <b>{risk}</b></div>
          <div>Grade: <b>{grade}°</b></div>
        </div>
        """,
        "style": {"backgroundColor": "rgba(5,10,20,0.92)", "color": "white"},
    }

    deck = pdk.Deck(
        layers=[trail_layer, driven_layer, highlight_layer, vehicle_layer],
        views=[pdk.View(type="OrbitView", controller=True)],
        initial_view_state=view_state,
        map_style=None,
        tooltip=tooltip,
    )
    return deck


# ----------------------------
# UI Helpers
# ----------------------------
def chip(text: str, kind: str = "ok") -> str:
    cls = {"ok": "r4-chip r4-chip-ok", "warn": "r4-chip r4-chip-warn", "danger": "r4-chip r4-chip-danger"}.get(
        kind, "r4-chip"
    )
    return f"<span class='{cls}'>{text}</span>"


def mode_chip(mode: str) -> str:
    if mode == "CRUISE":
        return chip("CRUISE", "ok")
    if mode == "CAUTIOUS":
        return chip("CAUTIOUS", "warn")
    if mode == "CRAWL":
        return chip("CRAWL", "warn")
    if mode == "STOP_SAFE":
        return chip("STOP SAFE", "danger")
    return chip(mode, "ok")


# ----------------------------
# Header
# ----------------------------
st.markdown("<div class='r4-title'>RAVEN 4 · AUTONOMY KERNEL COCKPIT</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='r4-sub'>Deterministic safety layer · 3D trail · human-gated actions · full audit log</div>",
    unsafe_allow_html=True,
)

# ----------------------------
# Layout
# ----------------------------
left_col, mid_col, right_col = st.columns([1.65, 1.15, 1.10], gap="large")

# ============================
# Right: controls
# ============================
with right_col:
    st.markdown("<div class='r4-card'>", unsafe_allow_html=True)
    st.markdown("### Environment & Envelope")

    preset = st.selectbox("Scenario", ["Mountain ridge", "Forest trail", "Desert wash"], index=0)
    weather = st.selectbox("Weather / traction", ["Dry", "Wet", "Snow / ice", "Dusty"], index=0)

    weather_factor = {"Dry": 0.10, "Wet": 0.25, "Snow / ice": 0.45, "Dusty": 0.20}[weather]

    risk_appetite = st.slider(
        "Driver risk appetite",
        0.0,
        1.0,
        0.40,
        help="0.0 = ultra conservative envelope, 1.0 = more aggressive envelope.",
    )

    clarity_floor = st.slider(
        "Stability attention floor",
        20.0,
        80.0,
        45.0,
        help="Below this stability, RAVEN always requests human attention.",
    )

    params = {"risk_appetite": float(risk_appetite), "weather_factor": float(weather_factor), "clarity_floor": float(clarity_floor)}

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### Simulation")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ Pulse 1 tick", use_container_width=True):
            # if scenario changed but user didn’t reset, keep current run; explicit reset is separate
            step_raven(params)
    with c2:
        if st.button("⟳ Pulse 10 ticks", use_container_width=True):
            for _ in range(10):
                step_raven(params)

    st.markdown("<div class='small-muted'>Each tick ≈ 1–2 seconds of trail time (synthetic).</div>", unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Autoplay (simple, deterministic: just a loop on this run)
    hz = st.slider("Autoplay rate (Hz)", 1.0, 20.0, float(st.session_state.r4["autoplay_hz"]), 0.5)
    st.session_state.r4["autoplay_hz"] = float(hz)
    autoplay_ticks = st.number_input("Autoplay ticks (per run)", min_value=5, max_value=400, value=60, step=5)

    c3, c4 = st.columns(2)
    with c3:
        if st.button("▶ Autoplay", use_container_width=True):
            # Don’t run through a human-lock; stop immediately when gate triggers.
            dt = 1.0 / max(0.1, float(hz))
            for _ in range(int(autoplay_ticks)):
                if st.session_state.r4["human_lock_active"]:
                    break
                step_raven(params)
                time.sleep(dt)
            st.rerun()
    with c4:
        if st.button("⏸ Stop", use_container_width=True):
            st.session_state.r4["paused"] = True

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Reset / regen
    if st.button("♻ Reset scenario + new run", use_container_width=True):
        reset_run(preset)
        st.rerun()

    # Audit download
    st.download_button(
        "⬇ Download audit NDJSON (this run)",
        data=audit_bytes_for_download(),
        file_name=f"raven4_{current_run_id()}.ndjson",
        mime="application/x-ndjson",
        use_container_width=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================
# Left: 3D cockpit + HUD
# ============================
with left_col:
    s = st.session_state.r4
    terrain = s["terrain"]
    v = s["vehicle"]
    decisions: List[DecisionEvent] = s["decisions"]

    # Scrub timeline (lets you replay a decision visually without changing state)
    max_tick = max(0, s["tick"])
    focus_tick = st.slider("Timeline scrub (visual focus only)", 0, max(1, max_tick), int(min(s["scrub_tick"], max_tick)))
    s["scrub_tick"] = int(focus_tick)

    deck = build_deck_scene(terrain=terrain, vehicle=v, path=s["path"], focus_tick=focus_tick if focus_tick > 0 else None)
    st.pydeck_chart(deck, use_container_width=True)

    # HUD
    st.markdown("<div class='r4-card'>", unsafe_allow_html=True)
    top = st.columns([1.2, 1.0, 1.0, 1.0, 1.0])
    with top[0]:
        st.markdown("<div class='r4-metric-label'>Mode</div>", unsafe_allow_html=True)
        st.markdown(mode_chip(v.mode), unsafe_allow_html=True)
    with top[1]:
        st.markdown("<div class='r4-metric-label'>Speed</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='r4-metric-value'>{v.speed_mps * 3.6:.1f} km/h</div>", unsafe_allow_html=True)
    with top[2]:
        st.markdown("<div class='r4-metric-label'>Stability</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='r4-metric-value'>{v.stability_index:.1f}</div>", unsafe_allow_html=True)
    with top[3]:
        st.markdown("<div class='r4-metric-label'>Drift</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='r4-metric-value'>{v.drift_score:.1f}</div>", unsafe_allow_html=True)
    with top[4]:
        st.markdown("<div class='r4-metric-label'>Grade</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='r4-metric-value'>{v.grade_deg:.1f}°</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='small-muted'>3D scene is cartesian (not map). Hover trail segments for risk + grade. "
        "Scrub slider highlights a segment without altering simulation state.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ============================
# Mid: Decision feed + human gate
# ============================
with mid_col:
    s = st.session_state.r4
    decisions: List[DecisionEvent] = s["decisions"]

    st.markdown("<div class='r4-card'>", unsafe_allow_html=True)
    st.markdown("### Decision feed · deterministic audit")

    if not decisions:
        st.markdown("<div class='small-muted'>No decisions yet. Pulse or autoplay to see RAVEN operate.</div>", unsafe_allow_html=True)
    else:
        # Human gate panel (locks on first human-required event until resolved)
        if s["human_lock_active"] and s["human_lock_index"] is not None and 0 <= s["human_lock_index"] < len(decisions):
            ev = decisions[s["human_lock_index"]]
            st.markdown(f"{chip('HUMAN GATE · REQUIRED', 'danger')}", unsafe_allow_html=True)
            st.markdown(
                f"**Tick {ev.tick} · {ev.action}**  \n"
                f"{ev.reason}  \n"
                f"Grade: `{ev.grade_deg:.1f}°` · Drift: `{ev.drift_score:.1f}` · Stability: `{ev.stability_index:.1f}`"
            )

            # Show thresholds that tripped it (inspectable)
            with st.expander("Show thresholds (inspectable envelope)"):
                st.json(ev.thresholds)

            a, b, c = st.columns(3)
            with a:
                if st.button("Approve", use_container_width=True):
                    decisions[s["human_lock_index"]].human_override = "APPROVED"
                    s["human_lock_active"] = False
                    s["human_lock_index"] = None
                    st.session_state.r4 = s
                    st.rerun()
            with b:
                if st.button("Deny", use_container_width=True):
                    decisions[s["human_lock_index"]].human_override = "DENIED"
                    # hard force safe stop
                    v = st.session_state.r4["vehicle"]
                    v.mode = "STOP_SAFE"
                    v.speed_mps = 0.0
                    st.session_state.r4["vehicle"] = v
                    s["human_lock_active"] = False
                    s["human_lock_index"] = None
                    st.session_state.r4 = s
                    st.rerun()
            with c:
                if st.button("Hold", use_container_width=True):
                    st.info("Holding: simulation can continue manually, but the gate remains unresolved.")

            st.markdown("<hr/>", unsafe_allow_html=True)

        # Recent events
        recent = decisions[-14:][::-1]
        for ev in recent:
            badge = ""
            if ev.human_required:
                if ev.human_override == "PENDING":
                    badge = " · `pending`"
                elif ev.human_override == "APPROVED":
                    badge = " · `approved`"
                elif ev.human_override == "DENIED":
                    badge = " · `denied`"

            # severity hint
            sev = "ok"
            if ev.action in ("CAUTIOUS",):
                sev = "warn"
            if ev.action in ("CRAWL", "STOP_SAFE"):
                sev = "danger"

            st.markdown(
                f"{chip(ev.action, sev)} "
                f"**Tick {ev.tick}{badge}**  \n"
                f"<span class='small-muted'>{ev.reason}<br>"
                f"Grade {ev.grade_deg:.1f}° · Drift {ev.drift_score:.1f} · Stability {ev.stability_index:.1f}</span>",
                unsafe_allow_html=True,
            )
            st.markdown("<hr/>", unsafe_allow_html=True)

        # Export decision table
        df = pd.DataFrame([asdict(x) for x in decisions])
        with st.expander("Decision table (inspectable)"):
            show_cols = ["tick", "seg_i", "action", "grade_deg", "drift_score", "stability_index", "human_required", "human_override", "reason"]
            st.dataframe(df[show_cols], use_container_width=True, height=320)

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Footer: “masterclass” notes baked into UI (short + sharp)
# ----------------------------
st.markdown(
    f"""
<div class='r4-card'>
  <div class='r4-metric-label'>What makes this credible</div>
  <div class='small-muted' style='margin-top:0.25rem;'>
    <b>Deterministic physics-to-policy split:</b> scores are math; thresholds are adjustable envelope. <br>
    <b>Hysteresis:</b> prevents mode thrash. <br>
    <b>Human gate:</b> STOP_SAFE/CRAWL + low-stability always request attention. <br>
    <b>Auditability:</b> every tick logs inputs, metrics, thresholds, and action (NDJSON). <br>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
