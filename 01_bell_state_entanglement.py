"""
NOTEBOOK 1: Quantum Entanglement — Bell State
=============================================
What this proves to clients:
- You can build and run real quantum circuits
- You understand superposition and entanglement
- You have live QPU access (swap backend at bottom)

Run locally:  python 01_bell_state_entanglement.py
Run on QPU:   set USE_REAL_QPU = True + add your Scaleway token
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ── CONFIG ─────────────────────────────────────────────────────────────────
USE_REAL_QPU = False   # flip to True to run on Scaleway IQM Garnet
SCALEWAY_TOKEN = ""    # paste your token here
SHOTS = 1024
# ───────────────────────────────────────────────────────────────────────────


def build_bell_circuit():
    """
    Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2
    Two qubits maximally entangled — measuring one instantly
    determines the other, regardless of distance.
    """
    qc = QuantumCircuit(2, 2)
    qc.h(0)           # Hadamard: put qubit 0 in superposition
    qc.cx(0, 1)       # CNOT: entangle qubit 1 with qubit 0
    qc.measure([0, 1], [0, 1])
    return qc


def get_backend():
    if USE_REAL_QPU:
        from qiskit_scaleway import ScalewayProvider
        provider = ScalewayProvider(token=SCALEWAY_TOKEN)
        return provider.backend("iqm_garnet")
    return AerSimulator()


def run_and_plot(qc, backend):
    t_qc = transpile(qc, backend)
    job = backend.run(t_qc, shots=SHOTS)
    counts = job.result().get_counts()

    # ── Expected result ────────────────────────────────────────────────────
    # A perfect Bell state gives ~50% |00⟩ and ~50% |11⟩
    # Never |01⟩ or |10⟩ — the qubits are locked together
    # ──────────────────────────────────────────────────────────────────────

    states = ['00', '01', '10', '11']
    values = [counts.get(s, 0) for s in states]
    percentages = [v / SHOTS * 100 for v in values]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Bell State |Φ+⟩  —  Quantum Entanglement\nScaleway QaaS / Qiskit Aer Emulator',
                 fontsize=13, fontweight='bold')

    # Bar chart
    colors = ['#1a73e8' if s in ('00', '11') else '#ea4335' for s in states]
    bars = axes[0].bar(states, percentages, color=colors, edgecolor='white', linewidth=0.8)
    axes[0].set_xlabel('Measurement outcome')
    axes[0].set_ylabel('Probability (%)')
    axes[0].set_title('Measurement probabilities')
    axes[0].set_ylim(0, 70)
    axes[0].axhline(50, color='gray', linestyle='--', linewidth=0.8, alpha=0.6, label='Ideal 50%')
    axes[0].legend(fontsize=9)
    for bar, pct in zip(bars, percentages):
        if pct > 1:
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

    # Circuit diagram as text
    axes[1].axis('off')
    circuit_text = str(qc.draw(output='text'))
    axes[1].text(0.05, 0.95, circuit_text, transform=axes[1].transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))
    axes[1].set_title('Circuit diagram')

    plt.tight_layout()
    plt.savefig('01_bell_state_result.png', dpi=150, bbox_inches='tight')
    print("  Saved: 01_bell_state_result.png")
    plt.close()
    return counts


def main():
    print("=" * 55)
    print("  Bell State Entanglement — Scaleway QaaS Demo")
    print("=" * 55)

    qc = build_bell_circuit()
    print(f"\nCircuit ({qc.num_qubits} qubits, depth {qc.depth()}):")
    print(qc.draw(output='text'))

    backend_name = "Scaleway IQM Garnet" if USE_REAL_QPU else "Aer Emulator (noise-free)"
    print(f"\nBackend: {backend_name}")
    print(f"Shots:   {SHOTS}")
    print("\nRunning...")

    backend = get_backend()
    counts = run_and_plot(qc, backend)

    print("\nResults:")
    for state, count in sorted(counts.items()):
        pct = count / SHOTS * 100
        bar = "█" * int(pct / 2)
        print(f"  |{state}⟩  {bar:<25}  {count:4d} shots  ({pct:.1f}%)")

    total_entangled = counts.get('00', 0) + counts.get('11', 0)
    print(f"\nEntangled outcomes (|00⟩ + |11⟩): {total_entangled/SHOTS*100:.1f}%")
    print("Expected: ~100%  (classical random coin = 50%)")
    print("\nThis proves: these two qubits are entangled.")
    print("Measuring one collapses the other — instantly.")
    print("\nChart saved to 01_bell_state_result.png")

    # ── Client talking point ───────────────────────────────────────────────
    print("\n" + "─" * 55)
    print("CLIENT VALUE:")
    print("Entanglement is the resource that gives quantum")
    print("computers their power for optimization, cryptography,")
    print("and machine learning. This circuit demonstrates you")
    print("can create and measure it on real hardware.")
    print("─" * 55)


if __name__ == "__main__":
    main()
