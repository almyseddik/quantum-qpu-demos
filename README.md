# Quantum Computing Portfolio — Scaleway QaaS

Real quantum circuits executed on Scaleway's European quantum cloud.

## Notebooks

| # | File | What it shows | QPU backend |
|---|------|--------------|-------------|
| 1 | `01_bell_state_entanglement.py` | Quantum entanglement — the foundation of all QC | IQM Garnet / Emulator |
| 2 | `02_qaoa_vehicle_routing.py` | QAOA solving a real logistics routing problem | IQM Garnet 20Q |
| 3 | `03_grover_search.py` | Grover's algorithm — quadratic speedup over classical search | IQM Emerald 54Q |

## Setup

```bash
pip install qiskit qiskit-aer pennylane matplotlib numpy
```

To run on real Scaleway QPU instead of emulator, replace the backend line with:
```python
from qiskit_scaleway import ScalewayProvider
backend = ScalewayProvider(token="YOUR_TOKEN").backend("iqm_garnet")
```

## About

These circuits run on Scaleway's quantum cloud: IQM Garnet (20Q), IQM Emerald (54Q),
AQT IBEX (12Q trapped-ion), Pasqal (neutral-atom), and NVIDIA Blackwell emulator (38Q noiseless).

Contact: @simokx | Available for quantum consulting projects
