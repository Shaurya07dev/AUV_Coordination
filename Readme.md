# Cloud-Edge AUV Swarm Digital Twin - Walkthrough

This guide explains how to run the prototype 3-Layer Digital Twin for AUV Swarms.

## 1. Installation

Ensure you have Python 3.9+ installed.

```bash
pip install -r requirements.txt
```

**Note:** If numpy installation is slow, try installing a pre-built binary: `pip install numpy`.

## 2. Running the System

You need to run these components in separate terminals:

### Terminal 1: Cloud Dashboard (Digital Twin)
Starts the 3D visualization.

```bash
streamlit run src/cloud/dashboard.py
```

### Terminal 2: Edge Dispatcher & Simulation
Runs the simulation loop which communicates with the Edge logic.

**Verify Simulation First:**
```bash
python -m experiments.baseline_random
```

**Run Full Integration (Coming Soon in Phase 4):**
```bash
# python src/main.py
```

## 3. Verifying Intelligence (Edge Layer)

Run the unit test for the Consensus-Based Bundle Algorithm (CBBA):

```bash
python -m experiments.test_consensus
```

**Expected Output:**
- `auv_1` (Low Battery) should be assigned to a `STATION`.
- `auv_0` (High Battery) should NOT be assigned.

## 4. Codebase Structure

*   **src/environment**: Local Layer. `auv_swarm_env.py` (3D Physics).
*   **src/edge**: Edge Layer. `dispatcher.py` (Rule-based CBBA), `mqtt_bridge.py`.
*   **src/cloud**: Cloud Layer. `dashboard.py` (Streamlit), `mappo_stub.py` (AI).
*   **experiments**: Verification scripts.