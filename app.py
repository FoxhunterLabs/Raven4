# ================================================================
# RAVEN 4 · Deterministic Autonomy Kernel
# Human-gated ONLY on dangerous conditions
# ================================================================

import math
import time
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict

import numpy as np
import streamlit as st
import pydeck as pdk

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="RAVEN 4 · Autonomy Kernel", layout="wide")

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
    roughness: float
    edge_exposure: float
    traction_coeff: float
    risk: float


@dataclass
class VehicleState:
    x: float
    y: float
    z: float
    speed_mps: float
    stability_index: float
    drift_score: float
    grade_deg: float
    mode: str


@dataclass
class DecisionEvent:
    tick: int
    seg_i: int
    action: str
    reason: str
    drift_score: float
    stability_index: float
    grade_deg: float
    human_required: bool
    timestamp: float


# ----------------------------
# Audit logging
# ----------------------------
RUNS_DIR = Path("runs_raven4")
RUNS_DIR.mkdir(exist_ok=True)

def run_id():
    if "run_id" not in st.session_state:
        st.session_state.run_id = time.strftime("%Y%m%dT%H%M%S")
    return st.session_state.run_id

def audit(rec: Dict):
    with open(RUNS_DIR / f"{run_id()}.ndjson", "a") as f:
        f.write(json.dumps(rec) + "\n")

# ----------------------------
# Terrain
# ----------------------------
def generate_terrain(n=160):
    rng = np.random.default_rng(42)
    x = y = z = 0.0
    out = []

    for i in range(n):
        t = i / n
        slope = math.sin(t * 2 * math.pi) * 15
        z += math.tan(math.radians(slope))
        x += 1.0

        rough = rng.uniform(0.2, 0.6)
        edge = rng.uniform(0.1, 0.6)
        traction = rng.uniform(0.4, 0.9)

        risk = min(1.0, 0.4 * rough + 0.4 * edge + 0.2 * abs(slope) / 18)

        out.append(TerrainSegment(
            i=i, x=x, y=y, z=z,
            slope_deg=slope,
            roughness=rough,
            edge_exposure=edge,
            traction_coeff=traction,
            risk=risk
        ))
    return out

# ----------------------------
# Scoring
# ----------------------------
def evaluate(seg: TerrainSegment, v: VehicleState):
    speed_factor = min(1.0, v.speed_mps / 15.0)

    drift = (
        0.4 * abs(seg.slope_deg) / 18 +
        0.3 * seg.roughness +
        0.3 * (1 - seg.traction_coeff)
    ) * 100

    stability = max(0.0, 100 - drift * (0.6 + 0.4 * speed_factor))

    return drift, stability, seg.slope_deg

# ----------------------------
# Decision logic (danger-focused)
# ----------------------------
def choose_action(drift, stability, grade):
    if drift > 85 or stability < 30 or abs(grade) > 20:
        return "STOP_SAFE", "outside hard safety envelope"
    if drift > 60 or stability < 50 or abs(grade) > 14:
        return "CRAWL", "high terrain risk"
    if drift > 40 or stability < 65 or abs(grade) > 10:
        return "CAUTIOUS", "moderate terrain risk"
    return "CRUISE", "stable window"

# ----------------------------
# Init
# ----------------------------
def init():
    terrain = generate_terrain()
    st.session_state.r4 = {
        "tick": 0,
        "terrain": terrain,
        "vehicle": VehicleState(
            x=terrain[0].x, y=terrain[0].y, z=terrain[0].z,
            speed_mps=0.0,
            stability_index=100.0,
            drift_score=0.0,
            grade_deg=0.0,
            mode="HOLD"
        ),
        "human_lock": False,
        "decisions": []
    }

if "r4" not in st.session_state:
    init()

# ----------------------------
# Step
# ----------------------------
def step():
    s = st.session_state.r4
    s["tick"] += 1
    i = min(len(s["terrain"]) - 1, s["tick"])
    seg = s["terrain"][i]
    v = s["vehicle"]

    drift, stability, grade = evaluate(seg, v)
    action, reason = choose_action(drift, stability, grade)

    human_required = action == "STOP_SAFE"

    if human_required:
        s["human_lock"] = True
        v.speed_mps = 0.0
        v.mode = "STOP_SAFE"
    else:
        target = {"CRUISE":12,"CAUTIOUS":6,"CRAWL":2}[action]
        v.speed_mps += 0.25 * (target - v.speed_mps)
        v.mode = action

    v.x, v.y, v.z = seg.x, seg.y, seg.z
    v.drift_score = drift
    v.stability_index = stability
    v.grade_deg = grade

    ev = DecisionEvent(
        tick=s["tick"],
        seg_i=i,
        action=action,
        reason=reason,
        drift_score=drift,
        stability_index=stability,
        grade_deg=grade,
        human_required=human_required,
        timestamp=time.time()
    )
    s["decisions"].append(ev)
    audit(asdict(ev))

# ----------------------------
# UI
# ----------------------------
st.title("RAVEN 4 · Deterministic Autonomy Kernel")

left, right = st.columns([2,1])
s = st.session_state.r4
v = s["vehicle"]

with left:
    trail = [{
        "src":[s["terrain"][i-1].x, s["terrain"][i-1].y, s["terrain"][i-1].z],
        "dst":[t.x,t.y,t.z],
        "color":[255*t.risk,200*(1-t.risk),80]
    } for i,t in enumerate(s["terrain"]) if i>0]

    st.pydeck_chart(pdk.Deck(
        layers=[
            pdk.Layer("LineLayer", trail, get_source_position="src", get_target_position="dst",
                      get_color="color", width_scale=2),
            pdk.Layer("ScatterplotLayer",
                      [{"pos":[v.x,v.y,v.z]}],
                      get_position="pos", get_radius=4,
                      get_fill_color=[0,224,164])
        ],
        initial_view_state=pdk.ViewState(target=[v.x,v.y,v.z], zoom=1, pitch=55),
        map_style=None
    ))

with right:
    st.markdown(f"**Mode:** {v.mode}")
    st.markdown(f"**Speed:** {v.speed_mps*3.6:.1f} km/h")
    st.markdown(f"**Stability:** {v.stability_index:.1f}")
    st.markdown(f"**Drift:** {v.drift_score:.1f}")

    if st.button("▶ Step"):
        step()

    if s["human_lock"]:
        st.error("⚠️ RAVEN entered STOP_SAFE\n\nHuman attention required.")
        if st.button("Acknowledge & continue"):
            s["human_lock"] = False
            st.rerun()
