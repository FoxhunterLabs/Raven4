# RAVEN 4 · Deterministic Autonomy Kernel (Demo)

RAVEN 4 is a **deterministic, safety-oriented autonomy kernel demonstrator** with a real-time 3D cockpit.  
It shows how transparent risk scoring, explicit safety envelopes, and **human supervision focused only on genuine danger** can be composed into a practical autonomy layer—without machine learning or black boxes.

This project is a **systems demo**, not a production vehicle controller.

---

## Design Philosophy

RAVEN 4 is built around one core idea:

> **Humans supervise danger, not complexity.**

The system is designed to handle rough terrain, uncertainty, and degraded conditions autonomously.  
Human attention is requested **only when the autonomy layer determines it has exited its safe operating envelope**.

This avoids operator overload while preserving meaningful human control.

---

## What RAVEN 4 Demonstrates

### 1. Synthetic 3D World Model
- Procedurally generated off-road trails
- Explicit **x, y, z** geometry with physically consistent grade
- Per-segment terrain attributes:
  - slope (grade)
  - roughness
  - edge exposure
  - traction coefficient
- Risk visualization along the trail

---

### 2. Deterministic Risk Scoring (No ML)
For each terrain segment, RAVEN computes:
- **Drift score** (0–100, higher = worse)
- **Stability index** (0–100, higher = better)

Inputs include:
- grade
- traction
- roughness
- edge exposure
- vehicle speed

All scoring is explicit math. No learned behavior. No hidden state.

---

### 3. Policy & Mode Selection
RAVEN maps risk into discrete operating modes:

- `CRUISE` – normal autonomous operation
- `CAUTIOUS` – reduced speed, increased margin
- `CRAWL` – low-speed traversal of high-risk terrain
- `STOP_SAFE` – autonomy confidence lost

Mode transitions are deterministic and inspectable.

---

### 4. Human-in-the-Loop (Danger-Only Gating)

Human input is required **only** when:

- The system enters `STOP_SAFE`, indicating it has exceeded its hard safety envelope

Human input is **not required** for:
- `CAUTIOUS`
- `CRAWL`
- rough or degraded terrain
- sustained low-speed risk

This mirrors real supervisory autonomy systems:  
operators intervene when autonomy confidence is lost, not on every difficult maneuver.

---

### 5. Auditability
Every simulation tick writes a structured NDJSON record:
- terrain inputs
- vehicle state
- computed risk metrics
- selected mode
- whether human attention was required

Logs are downloadable and suitable for replay or offline inspection.

---

## Visualization & UI

- 3D trail rendered with PyDeck (deck.gl)
- Risk-colored terrain ribbon
- Live vehicle position
- Cockpit HUD:
  - mode
  - speed
  - stability
  - drift
  - grade
- Human alert appears **only for true safety violations**

The UI is designed for **inspection and reasoning**, not presentation.

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
pip install streamlit pydeck numpy
