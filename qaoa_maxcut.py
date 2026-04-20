#!/usr/bin/env python3
"""Tutorial-friendly QAOA helpers for solving Max-Cut with Qiskit."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize


EXAMPLE_EDGES = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]


@dataclass
class OptimizationResult:
    x: np.ndarray
    expected_cut: float
    counts: dict[str, int]
    cut_distribution: dict[str, int]
    objective_history: list[float]


@dataclass
class MaxCutSolution:
    graph: nx.Graph
    result: OptimizationResult
    exact_cut: int
    exact_assignments: list[str]


@dataclass(frozen=True)
class SimpleNoiseModelConfig:
    single_qubit_gate_error: float = 0.001
    two_qubit_gate_error: float = 0.01
    readout_error: float = 0.02


@dataclass(frozen=True)
class FakeBackendConfig:
    backend_name: str = "FakeManilaV2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve Max-Cut with QAOA in Qiskit.")
    parser.add_argument(
        "--edges",
        type=str,
        default=",".join(f"{u}-{v}" for u, v in EXAMPLE_EDGES),
        help='Comma-separated edges, for example "0-1,1-2,2-3,3-0".',
    )
    parser.add_argument("--layers", type=int, default=2, help="Number of QAOA layers p.")
    parser.add_argument(
        "--shots",
        type=int,
        default=2048,
        help="Shots used during optimization and the final sampling run.",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=8,
        help="Random restarts for the classical optimizer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for the optimizer and the sampler.",
    )
    return parser.parse_args()


def parse_edges(edge_text: str) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for token in edge_text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            left, right = token.split("-")
            u, v = int(left), int(right)
        except ValueError as exc:
            raise ValueError(f"Invalid edge format: {token!r}. Expected 'u-v'.") from exc
        if u == v:
            raise ValueError(f"Self-loops are not supported: {token!r}")
        edges.append((u, v))

    if not edges:
        raise ValueError("At least one edge is required.")
    return edges


def build_graph(edges: Iterable[tuple[int, int]]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return nx.convert_node_labels_to_integers(graph, ordering="sorted")


def create_example_graph() -> nx.Graph:
    return build_graph(EXAMPLE_EDGES)


def bitstring_to_assignment(bitstring: str, num_nodes: int) -> str:
    # Qiskit returns bitstrings as c[n-1]...c[0], so reverse to align with node order.
    ordered = bitstring[::-1]
    if len(ordered) != num_nodes:
        raise ValueError(f"Expected {num_nodes} bits, received {bitstring!r}")
    return ordered


def assignment_to_partition(assignment: str) -> tuple[list[int], list[int]]:
    left = [index for index, bit in enumerate(assignment) if bit == "0"]
    right = [index for index, bit in enumerate(assignment) if bit == "1"]
    return left, right


def cut_value(assignment: str, graph: nx.Graph) -> int:
    return sum(assignment[u] != assignment[v] for u, v in graph.edges())


def brute_force_maxcut(graph: nx.Graph) -> tuple[int, list[str]]:
    best_value = -1
    best_assignments: list[str] = []
    for value in range(1 << graph.number_of_nodes()):
        assignment = format(value, f"0{graph.number_of_nodes()}b")
        current = cut_value(assignment, graph)
        if current > best_value:
            best_value = current
            best_assignments = [assignment]
        elif current == best_value:
            best_assignments.append(assignment)
    return best_value, best_assignments


def _edge_pauli_label(num_qubits: int, u: int, v: int) -> str:
    label = ["I"] * num_qubits
    label[num_qubits - 1 - u] = "Z"
    label[num_qubits - 1 - v] = "Z"
    return "".join(label)


def build_maxcut_hamiltonian(graph: nx.Graph) -> SparsePauliOp:
    num_qubits = graph.number_of_nodes()
    pauli_terms: list[tuple[str, float]] = [("I" * num_qubits, 0.0)]

    for u, v in graph.edges():
        pauli_terms.append(("I" * num_qubits, 0.5))
        pauli_terms.append((_edge_pauli_label(num_qubits, u, v), -0.5))

    return SparsePauliOp.from_list(pauli_terms).simplify()


def build_cost_layer(circuit: QuantumCircuit, graph: nx.Graph, gamma) -> None:
    for u, v in graph.edges():
        circuit.cx(u, v)
        circuit.rz(2 * gamma, v)
        circuit.cx(u, v)


def build_mixer_layer(circuit: QuantumCircuit, graph: nx.Graph, beta) -> None:
    for qubit in graph.nodes():
        circuit.rx(2 * beta, qubit)


def build_qaoa_circuit(graph: nx.Graph, layers: int) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    num_qubits = graph.number_of_nodes()
    gamma = ParameterVector("gamma", layers)
    beta = ParameterVector("beta", layers)

    circuit = QuantumCircuit(num_qubits)
    circuit.h(range(num_qubits))

    for layer in range(layers):
        build_cost_layer(circuit, graph, gamma[layer])
        build_mixer_layer(circuit, graph, beta[layer])

    circuit.measure_all()
    return circuit, gamma, beta


def build_simple_noise_model(config: SimpleNoiseModelConfig | None = None):
    if config is None:
        config = SimpleNoiseModelConfig()

    try:
        from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    except ImportError as exc:
        raise ImportError(
            "Noisy simulations require qiskit-aer. Install it with `pip install qiskit-aer`."
        ) from exc

    noise_model = NoiseModel()

    if config.single_qubit_gate_error > 0.0:
        single_qubit_error = depolarizing_error(config.single_qubit_gate_error, 1)
        noise_model.add_all_qubit_quantum_error(single_qubit_error, ["h", "rx", "rz"])

    if config.two_qubit_gate_error > 0.0:
        two_qubit_error = depolarizing_error(config.two_qubit_gate_error, 2)
        noise_model.add_all_qubit_quantum_error(two_qubit_error, ["cx"])

    if config.readout_error > 0.0:
        readout_error = ReadoutError(
            [
                [1.0 - config.readout_error, config.readout_error],
                [config.readout_error, 1.0 - config.readout_error],
            ]
        )
        noise_model.add_all_qubit_readout_error(readout_error)

    return noise_model


def build_fake_backend(config: FakeBackendConfig | None = None):
    if config is None:
        config = FakeBackendConfig()

    try:
        import qiskit_ibm_runtime.fake_provider as fake_provider
    except ImportError as exc:
        raise ImportError(
            "Fake IBM backend experiments require qiskit-ibm-runtime. Install it with `pip install qiskit-ibm-runtime`."
        ) from exc

    try:
        backend_class = getattr(fake_provider, config.backend_name)
    except AttributeError as exc:
        raise ValueError(
            f"Unknown fake backend {config.backend_name!r}. "
            "Use a backend class from qiskit_ibm_runtime.fake_provider, "
            "for example 'FakeManilaV2' or 'FakeQuitoV2'."
        ) from exc

    return backend_class()


def build_fake_backend_simulator(config: FakeBackendConfig | None = None):
    fake_backend = build_fake_backend(config)
    try:
        from qiskit_aer import AerSimulator
    except ImportError as exc:
        raise ImportError(
            "Fake IBM backend experiments require qiskit-aer. Install it with `pip install qiskit-aer`."
        ) from exc

    simulator = AerSimulator.from_backend(fake_backend)
    return fake_backend, simulator


def sample_counts(
    circuit: QuantumCircuit,
    params: np.ndarray | list[float],
    shots: int = 2048,
    seed: int = 7,
    noise_model=None,
    simulator_backend=None,
) -> dict[str, int]:
    if simulator_backend is not None:
        bound_circuit = circuit.assign_parameters(params)
        compiled_circuit = transpile(bound_circuit, simulator_backend, optimization_level=0, seed_transpiler=seed)
        result = simulator_backend.run(compiled_circuit, shots=shots).result()
        return result.get_counts(0)

    if noise_model is not None:
        try:
            from qiskit_aer import AerSimulator
        except ImportError as exc:
            raise ImportError(
                "Noisy simulations require qiskit-aer. Install it with `pip install qiskit-aer`."
            ) from exc

        bound_circuit = circuit.assign_parameters(params)
        backend = AerSimulator(noise_model=noise_model, seed_simulator=seed)
        compiled_circuit = transpile(bound_circuit, backend, optimization_level=0, seed_transpiler=seed)
        result = backend.run(compiled_circuit, shots=shots).result()
        return result.get_counts(0)

    sampler = StatevectorSampler(default_shots=shots, seed=seed)
    return sampler.run([(circuit, params)]).result()[0].data.meas.get_counts()


def evaluate_counts(counts: dict[str, int], graph: nx.Graph) -> tuple[float, dict[str, int]]:
    total_shots = sum(counts.values())
    if total_shots == 0:
        raise ValueError("Sampler returned zero shots.")

    expected_cut = 0.0
    cut_distribution: dict[str, int] = {}
    for raw_bitstring, frequency in counts.items():
        assignment = bitstring_to_assignment(raw_bitstring, graph.number_of_nodes())
        value = cut_value(assignment, graph)
        expected_cut += value * frequency / total_shots
        cut_distribution[assignment] = frequency
    return expected_cut, cut_distribution


def rank_assignments(
    cut_distribution: dict[str, int],
    graph: nx.Graph,
    limit: int = 5,
) -> list[tuple[str, int, int]]:
    ranked = []
    for assignment, frequency in sorted(cut_distribution.items(), key=lambda item: item[1], reverse=True)[:limit]:
        ranked.append((assignment, frequency, cut_value(assignment, graph)))
    return ranked


def counts_to_probabilities(counts: dict[str, int], graph: nx.Graph) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        raise ValueError("Counts must contain at least one sample.")
    probabilities: dict[str, float] = {}
    for raw_bitstring, frequency in counts.items():
        assignment = bitstring_to_assignment(raw_bitstring, graph.number_of_nodes())
        probabilities[assignment] = frequency / total
    return probabilities


def optimal_solution_probability(
    counts: dict[str, int],
    graph: nx.Graph,
    optimal_assignments: Iterable[str] | None = None,
) -> float:
    if optimal_assignments is None:
        _, optimal_assignments = brute_force_maxcut(graph)

    probabilities = counts_to_probabilities(counts, graph)
    return sum(probabilities.get(assignment, 0.0) for assignment in optimal_assignments)


def all_bitstrings(num_bits: int) -> list[str]:
    return [format(value, f"0{num_bits}b") for value in range(1 << num_bits)]


def plot_sampling_distribution(
    counts: dict[str, int],
    graph: nx.Graph,
    ax=None,
    title: str = "Sampling distribution over all bitstrings",
):
    probabilities = counts_to_probabilities(counts, graph)
    labels = all_bitstrings(graph.number_of_nodes())
    values = [probabilities.get(assignment, 0.0) for assignment in labels]
    max_cut = max(cut_value(assignment, graph) for assignment in labels)
    colors = [
        "#DD8452" if cut_value(label, graph) == max_cut else "#4C72B0"
        for label in labels
    ]

    if ax is None:
        width = max(8, 0.45 * len(labels))
        _, ax = plt.subplots(figsize=(width, 4))

    ax.bar(labels, values, color=colors)
    ax.set_xlabel("Bitstring assignment")
    ax.set_ylabel("Sampling probability")
    ax.set_title(title)
    ax.set_ylim(0.0, max(values) * 1.15 if values else 1.0)
    ax.tick_params(axis="x", rotation=90)
    return ax


def optimize_qaoa(
    graph: nx.Graph,
    layers: int,
    shots: int,
    restarts: int,
    seed: int,
    noise_model=None,
    simulator_backend=None,
) -> OptimizationResult:
    circuit, _, _ = build_qaoa_circuit(graph, layers)
    rng = np.random.default_rng(seed)
    objective_history: list[float] = []

    def objective(params: np.ndarray) -> float:
        counts = sample_counts(
            circuit,
            params,
            shots=shots,
            seed=seed,
            noise_model=noise_model,
            simulator_backend=simulator_backend,
        )
        expected_cut, _ = evaluate_counts(counts, graph)
        value = -expected_cut
        objective_history.append(value)
        return value

    best = None
    bounds = [(0.0, np.pi)] * layers + [(0.0, np.pi / 2)] * layers

    for _ in range(restarts):
        initial = np.concatenate(
            [
                rng.uniform(0.0, np.pi, size=layers),
                rng.uniform(0.0, np.pi / 2, size=layers),
            ]
        )
        result = minimize(objective, initial, method="COBYLA", bounds=bounds)
        if best is None or result.fun < best.fun:
            best = result

    if best is None:
        raise RuntimeError("Optimization did not produce a result.")

    final_counts = sample_counts(
        circuit,
        best.x,
        shots=shots,
        seed=seed,
        noise_model=noise_model,
        simulator_backend=simulator_backend,
    )
    expected_cut, cut_distribution = evaluate_counts(final_counts, graph)
    return OptimizationResult(best.x, expected_cut, final_counts, cut_distribution, objective_history)


def solve_maxcut_instance(
    edges: Iterable[tuple[int, int]] | None = None,
    *,
    graph: nx.Graph | None = None,
    layers: int = 2,
    shots: int = 2048,
    restarts: int = 8,
    seed: int = 7,
    noise_model=None,
    simulator_backend=None,
) -> MaxCutSolution:
    if graph is None:
        if edges is None:
            raise ValueError("Provide either edges or graph.")
        graph = build_graph(edges)

    result = optimize_qaoa(
        graph=graph,
        layers=layers,
        shots=shots,
        restarts=restarts,
        seed=seed,
        noise_model=noise_model,
        simulator_backend=simulator_backend,
    )
    exact_cut, exact_assignments = brute_force_maxcut(graph)
    return MaxCutSolution(graph, result, exact_cut, exact_assignments)


def draw_graph(
    graph: nx.Graph,
    assignment: str | None = None,
    title: str | None = None,
    ax=None,
    seed: int = 7,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    positions = nx.spring_layout(graph, seed=seed)
    if assignment is None:
        node_colors = ["#1F6AA5"] * graph.number_of_nodes()
        font_color = "white"
        base_edges = list(graph.edges())
        cut_edges: list[tuple[int, int]] = []
    else:
        node_colors = ["#4C72B0" if bit == "0" else "#DD8452" for bit in assignment]
        font_color = "white"
        cut_edges = [(u, v) for u, v in graph.edges() if assignment[u] != assignment[v]]
        base_edges = [(u, v) for u, v in graph.edges() if assignment[u] == assignment[v]]

    nx.draw_networkx_nodes(
        graph,
        pos=positions,
        ax=ax,
        node_color=node_colors,
        node_size=1100,
        linewidths=1.5,
        edgecolors="#1f1f1f",
    )
    nx.draw_networkx_labels(
        graph,
        pos=positions,
        ax=ax,
        font_color=font_color,
        font_size=13,
        font_weight="bold",
    )
    nx.draw_networkx_edges(
        graph,
        pos=positions,
        ax=ax,
        edgelist=base_edges,
        edge_color="#4a4a4a",
        width=2.0,
    )
    if cut_edges:
        nx.draw_networkx_edges(
            graph,
            pos=positions,
            ax=ax,
            edgelist=cut_edges,
            edge_color="#C44E52",
            width=3.5,
        )
    if title:
        ax.set_title(title)
    ax.set_axis_off()
    return ax


def print_report(graph: nx.Graph, result: OptimizationResult, layers: int) -> None:
    exact_value, exact_assignments = brute_force_maxcut(graph)
    gamma_values = result.x[:layers]
    beta_values = result.x[layers:]
    ranked = rank_assignments(result.cut_distribution, graph)
    best_sample_assignment, _ = max(result.cut_distribution.items(), key=lambda item: item[1])
    best_sample_value = cut_value(best_sample_assignment, graph)

    print("=== Graph ===")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {list(graph.edges())}")
    print()
    print("=== QAOA Result ===")
    print(f"Layers p: {layers}")
    print(f"Expected cut value: {result.expected_cut:.4f}")
    print(f"gamma: {np.round(gamma_values, 6).tolist()}")
    print(f"beta:  {np.round(beta_values, 6).tolist()}")
    print()
    print("=== Top Sampled Assignments ===")
    for assignment, frequency, value in ranked:
        print(f"{assignment}  shots={frequency:<4d}  cut={value}")
    print()
    print("=== Best Sample vs Exact Optimum ===")
    print(f"Most frequent sample: {best_sample_assignment}  cut={best_sample_value}")
    print(f"Exact optimum cut: {exact_value}")
    print(f"Exact optimum assignments: {exact_assignments}")


def main() -> None:
    args = parse_args()
    solution = solve_maxcut_instance(
        edges=parse_edges(args.edges),
        layers=args.layers,
        shots=args.shots,
        restarts=args.restarts,
        seed=args.seed,
    )
    print_report(solution.graph, solution.result, args.layers)


if __name__ == "__main__":
    main()
