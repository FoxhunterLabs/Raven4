________________________________________
# RAVEN 4 · Deterministic Autonomy Kernel (Demo)

RAVEN 4 is a **deterministic, human-gated autonomy kernel demonstrator** with a real-time 3D cockpit.  
It shows how transparent safety logic, inspectable thresholds, and human oversight can be composed into a practical autonomy layer—without machine learning or black boxes.

This project is a **“brain-on-a-table” demo**, not a production autonomy stack.

---

## What RAVEN 4 Demonstrates

**RAVEN 4 focuses on system architecture and safety reasoning**, not perception or learning.

### Core ideas
- Deterministic physics → risk → policy (no learned behavior)
- Explicit safety envelopes with hysteresis (no mode thrash)
- Human-gated actions for high-risk states
- Full per-tick audit log (NDJSON)
- Operator-oriented 3D visualization

Everything the system uses to make decisions is visible, logged, and overrideable.

---

## System Overview

### 1. Synthetic World Model (3D)
- Procedurally generated off-road trails
- Explicit **x, y, z** geometry with physically consistent grade
- Terrain attributes per segment:
  - slope (grade)
  - roughness
  - edge exposure
  - traction coefficient
- Scenarios: Mountain ridge, Forest trail, Desert wash

### 2. Deterministic Risk Scoring
For each terrain segment, RAVEN computes:
- Drift score (0–100)
- Stability index (0–100)

Inputs include:
- grade
- traction
- roughness
- edge exposure
- vehicle speed
- weather factor

No learning, no inference—just math.

---

### 3. Policy & Safety Layer
RAVEN maps scores into discrete modes:

- `CRUISE`
- `CAUTIOUS`
- `CRAWL`
- `STOP_SAFE`

Features:
- Adjustable risk appetite (moves thresholds, not physics)
- Hysteresis to prevent rapid mode switching
- Stability “clarity floor” that always requests human attention
- Mandatory human gating for `CRAWL` and `STOP_SAFE`

---

### 4. Human-in-the-Loop Control
When a human gate triggers:
- Simulation pauses on the first unsafe decision
- Operator can **Approve**, **Deny**, or **Hold**
- Deny forces an immediate safe stop

This is a first-class system constraint, not an afterthought.

---

### 5. Auditability
Every simulation tick writes a structured NDJSON record:
- Terrain inputs
- Vehicle state
- Computed metrics
- Thresholds used
- Action taken
- Human override status

Logs are downloadable and replayable.

---

## UI / Visualization

- 3D trail rendered with PyDeck (deck.gl)
- Risk-colored terrain ribbon
- Driven path overlay
- Live vehicle marker
- Cockpit HUD (mode, speed, drift, stability, grade)
- Timeline scrubber to visually inspect past decisions

The UI is designed for **inspection and interrogation**, not presentation.

---

## What This Is / Is Not

**This is:**
- A transparent autonomy kernel demo
- A safety-oriented control layer example
- A systems-level autonomy artifact

**This is not:**
- A perception stack
- A learned driving policy
- A production-ready autonomy system
- A claim about real-world vehicle performance

---

## Running Locally

### Requirements
```bash
python 3.10+
pip install streamlit pydeck numpy pandas
Run
streamlit run app.py
The app runs entirely locally. No external services required.
________________________________________
File Structure
.
├── app.py              # Full RAVEN 4 system (single-file demo)
├── runs_raven4/        # NDJSON audit logs (per run)
└── README.md
________________________________________
Design Philosophy
RAVEN 4 is built around a simple premise:
Autonomy earns trust through clarity, determinism, and human control—not opacity.
The goal is not to replace drivers or operators, but to build autonomy layers that:
•	explain themselves
•	fail safely
•	and always defer to human judgment when uncertainty rises
________________________________________
License / Use
MIT
________________________________________
If you are evaluating this system:
Everything RAVEN does is visible by design. That constraint is the point.
