"""
NOTEBOOK 3: Grover's Search Algorithm — Quantum Speedup
========================================================
What this proves to clients:
- Quantum computers can search N items in √N steps
- Classical computers need N/2 steps on average
- For 1 million items: classical = 500,000 checks, quantum = 1,000 checks
- This is directly applicable to: database search, fraud detection,
  drug candidate screening, financial portfolio search

Run locally:  python 03_grover_search.py
Run on QPU:   set USE_REAL_QPU = True + Scaleway token
"""

import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ── CONFIG ─────────────────────────────────────────────────────────────────
USE_REAL_QPU = False
SCALEWAY_TOKEN = ""
N_QUBITS = 4      # search space = 2^4 = 16 items
TARGET = "1011"   # the item we're looking for (binary string)
SHOTS = 2048
# ───────────────────────────────────────────────────────────────────────────


def classical_search(database, target):
    """
    Classical random search — average N/2 checks to find target.
    """
    checks = 0
    indices = list(range(len(database)))
    np.random.shuffle(indices)
    for idx in indices:
        checks += 1
        if database[idx] == target:
            return idx, checks
    return -1, checks


def grover_oracle(n, target):
    """
    Phase oracle: marks the target state with a phase flip (-1).
    Only the target state |target⟩ gets flipped.
    """
    qc = QuantumCircuit(n)
    target_bits = [int(b) for b in target]

    # Flip qubits where target bit is 0 (so MCNOT fires on |target⟩)
    for i, bit in enumerate(reversed(target_bits)):
        if bit == 0:
            qc.x(i)

    # Multi-controlled Z gate
    qc.h(n-1)
    qc.mcx(list(range(n-1)), n-1)
    qc.h(n-1)

    # Unflip
    for i, bit in enumerate(reversed(target_bits)):
        if bit == 0:
            qc.x(i)

    return qc


def grover_diffuser(n):
    """
    Grover diffusion operator: amplifies the marked state.
    Also called the 'inversion about the mean'.
    """
    qc = QuantumCircuit(n)
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n-1)
    qc.mcx(list(range(n-1)), n-1)
    qc.h(n-1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def build_grover_circuit(n, target, iterations=None):
    """
    Full Grover circuit with optimal number of iterations.
    Optimal iterations ≈ π/4 × √(N/M)
    where N = search space size, M = number of solutions
    """
    N = 2 ** n
    if iterations is None:
        iterations = max(1, int(np.round(np.pi / 4 * np.sqrt(N))))

    qc = QuantumCircuit(n, n)

    # Step 1: Create equal superposition of all states
    qc.h(range(n))

    # Step 2: Repeat oracle + diffuser
    oracle = grover_oracle(n, target)
    diffuser = grover_diffuser(n)

    for _ in range(iterations):
        qc = qc.compose(oracle)
        qc = qc.compose(diffuser)

    qc.measure(range(n), range(n))
    return qc, iterations


def get_backend():
    if USE_REAL_QPU:
        from qiskit_scaleway import ScalewayProvider
        return ScalewayProvider(token=SCALEWAY_TOKEN).backend("iqm_emerald")
    return AerSimulator()


def plot_results(counts, target, n_qubits, n_iterations,
                 quantum_checks, classical_avg):
    N = 2 ** n_qubits
    all_states = [format(i, f'0{n_qubits}b') for i in range(N)]
    probabilities = [counts.get(s, 0) / SHOTS * 100 for s in all_states]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Grover's Search Algorithm — Quantum Speedup\nScaleway QaaS Demo",
                 fontsize=13, fontweight='bold')

    # Probability distribution
    colors = ['#1565c0' if s == target else '#bbdefb' for s in all_states]
    bars = axes[0].bar(all_states, probabilities, color=colors, edgecolor='white')
    axes[0].set_xlabel('State (binary)')
    axes[0].set_ylabel('Probability (%)')
    axes[0].set_title(f'Measurement outcomes after {n_iterations} Grover iteration(s)\nTarget: |{target}⟩')
    axes[0].tick_params(axis='x', rotation=45)

    target_pct = counts.get(target, 0) / SHOTS * 100
    target_idx = all_states.index(target)
    axes[0].annotate(f'TARGET\n{target_pct:.1f}%',
                    xy=(target_idx, target_pct),
                    xytext=(target_idx + 1.5, target_pct * 0.85),
                    fontsize=9, color='#1565c0',
                    arrowprops=dict(arrowstyle='->', color='#1565c0'))

    uniform = 100 / N
    axes[0].axhline(uniform, color='gray', linestyle='--',
                   linewidth=0.8, alpha=0.6, label=f'Random: {uniform:.1f}%')
    axes[0].legend(fontsize=9)

    # Speedup comparison
    problem_sizes = [4, 8, 16, 64, 256, 1024, 4096, 16384]
    classical_checks = [n/2 for n in problem_sizes]
    quantum_checks_sizes = [np.ceil(np.pi/4 * np.sqrt(n)) for n in problem_sizes]

    axes[1].loglog(problem_sizes, classical_checks, 'r-o',
                  label='Classical (N/2)', linewidth=2, markersize=6)
    axes[1].loglog(problem_sizes, quantum_checks_sizes, 'b-s',
                  label='Grover (√N)', linewidth=2, markersize=6)

    current_n = 2 ** n_qubits
    axes[1].axvline(current_n, color='green', linestyle=':', alpha=0.7)
    axes[1].text(current_n * 1.1, classical_checks[3],
                f'This demo\n(N={current_n})', fontsize=8, color='green')

    axes[1].set_xlabel('Database size (N)')
    axes[1].set_ylabel('Queries needed')
    axes[1].set_title('Quantum vs Classical — scaling advantage')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    speedup_1M = 500000 / 1000
    axes[1].text(0.05, 0.15, f'At N=1,000,000:\nClassical: 500,000 checks\nQuantum: ~1,000 checks\nSpeedup: {speedup_1M:.0f}×',
                transform=axes[1].transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('03_grover_search_result.png', dpi=150, bbox_inches='tight')
    print("  Saved: 03_grover_search_result.png")
    plt.close()


def main():
    print("=" * 60)
    print("  Grover's Search Algorithm — Scaleway QaaS Demo")
    print("=" * 60)

    N = 2 ** N_QUBITS
    optimal_iters = max(1, int(np.round(np.pi / 4 * np.sqrt(N))))

    print(f"\nSearch space: {N} items (2^{N_QUBITS})")
    print(f"Target item:  |{TARGET}⟩  (item #{int(TARGET, 2)} of {N})")
    print(f"Grover iterations: {optimal_iters}  (optimal = π/4 × √N = {np.pi/4*np.sqrt(N):.2f})")

    # ── Classical baseline ─────────────────────────────────────────────────
    print("\n[1/2] Classical random search...")
    database = [format(i, f'0{N_QUBITS}b') for i in range(N)]
    trials = 500
    total_checks = 0
    for _ in range(trials):
        _, checks = classical_search(database, TARGET)
        total_checks += checks
    classical_avg = total_checks / trials
    print(f"      Average checks needed: {classical_avg:.1f}  (theory: N/2 = {N/2})")
    print(f"      Success rate: 100% (always finds it, just slow)")

    # ── Grover's algorithm ─────────────────────────────────────────────────
    print(f"\n[2/2] Grover's algorithm ({N_QUBITS} qubits, {SHOTS} shots)...")
    backend_name = "Scaleway IQM Emerald" if USE_REAL_QPU else "Aer Emulator"
    print(f"      Backend: {backend_name}")

    qc, iterations = build_grover_circuit(N_QUBITS, TARGET)
    print(f"      Circuit: {qc.num_qubits} qubits, depth {qc.depth()}, {iterations} iteration(s)")

    backend = get_backend()
    t0 = time.time()
    t_qc = transpile(qc, backend, optimization_level=1)
    job = backend.run(t_qc, shots=SHOTS)
    counts = job.result().get_counts()
    t_grover = time.time() - t0

    target_count = counts.get(TARGET, 0)
    target_prob = target_count / SHOTS * 100
    theoretical_prob = np.sin((2*iterations+1) * np.arcsin(1/np.sqrt(N)))**2 * 100
    quantum_checks = iterations  # Grover uses 1 oracle query per iteration

    print(f"      Target |{TARGET}⟩ found: {target_count}/{SHOTS} shots ({target_prob:.1f}%)")
    print(f"      Theoretical probability: {theoretical_prob:.1f}%")
    print(f"      Oracle queries used: {quantum_checks}  (classical avg: {classical_avg:.0f})")

    speedup = classical_avg / quantum_checks
    print(f"      Speedup: {speedup:.1f}×")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("RESULTS SUMMARY")
    print("─" * 60)
    print(f"  Search space:        {N} items")
    print(f"  Target:              |{TARGET}⟩")
    print(f"  Classical (avg):     {classical_avg:.0f} checks")
    print(f"  Grover queries:      {quantum_checks}")
    print(f"  Speedup this demo:   {speedup:.1f}×")
    print(f"  At N=1,000,000:      500,000 classical vs ~1,000 quantum")
    print(f"  At N=1,000,000,000:  500M classical vs ~31,623 quantum")

    plot_results(counts, TARGET, N_QUBITS, iterations,
                quantum_checks, classical_avg)

    print("\n" + "─" * 60)
    print("CLIENT VALUE:")
    print("Pharmaceutical companies screen millions of drug")
    print("candidates against protein targets. Banks scan billions")
    print("of transactions for fraud. Grover's algorithm applies")
    print("a quadratic speedup to any unstructured search problem.")
    print("This is why companies are investing in quantum NOW.")
    print("─" * 60)


if __name__ == "__main__":
    main()
