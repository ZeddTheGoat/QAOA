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
    iteration_history: list[float]


@dataclass
class MaxCutSolution:
    graph: nx.Graph
    result: OptimizationResult
    exact_cut: int
    exact_assignments: list[str]


@dataclass
class LayerSweepEntry:
    layers: int
    initial_params: np.ndarray | None
    result: OptimizationResult
    optimal_probability: float


@dataclass
class LayerSweepResult:
    graph: nx.Graph
    exact_cut: int
    exact_assignments: list[str]
    entries: list[LayerSweepEntry]


@dataclass(frozen=True)
class SGDConfig:
    steps: int = 40
    learning_rate: float = 0.12
    gradient_step: float = 0.1
    learning_rate_decay: float = 0.02


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
    parser.add_argument(
        "--optimizer",
        type=str.lower,
        default="cobyla",
        choices=["cobyla", "sgd"],
        help="Classical optimizer used for QAOA parameter search.",
    )
    parser.add_argument(
        "--sgd-steps",
        type=int,
        default=40,
        help="Number of SGD update steps when --optimizer SGD is used.",
    )
    parser.add_argument(
        "--sgd-learning-rate",
        type=float,
        default=0.12,
        help="Initial learning rate for SGD.",
    )
    parser.add_argument(
        "--sgd-gradient-step",
        type=float,
        default=0.1,
        help="Finite-difference step used to estimate SGD gradients.",
    )
    parser.add_argument(
        "--sgd-learning-rate-decay",
        type=float,
        default=0.02,
        help="Per-step learning-rate decay for SGD.",
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


def fourier_parameter_matrices(layers: int, modes: int) -> tuple[np.ndarray, np.ndarray]:
    if layers <= 0:
        raise ValueError("layers must be positive.")
    if modes <= 0:
        raise ValueError("modes must be positive.")
    if modes > layers:
        raise ValueError("modes cannot exceed layers.")

    layer_indices = np.arange(1, layers + 1, dtype=float)[:, None]
    mode_indices = np.arange(1, modes + 1, dtype=float)[None, :]
    angles = (mode_indices - 0.5) * (layer_indices - 0.5) * np.pi / layers
    return np.sin(angles), np.cos(angles)


def angles_to_fourier(
    gamma: np.ndarray | list[float],
    beta: np.ndarray | list[float],
    modes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    gamma_array = np.asarray(gamma, dtype=float)
    beta_array = np.asarray(beta, dtype=float)
    if gamma_array.ndim != 1 or beta_array.ndim != 1:
        raise ValueError("gamma and beta must be one-dimensional arrays.")
    if len(gamma_array) != len(beta_array):
        raise ValueError("gamma and beta must have the same length.")

    layers = len(gamma_array)
    if layers == 0:
        raise ValueError("At least one QAOA layer is required.")

    if modes is None:
        modes = layers
    if modes <= 0:
        raise ValueError("modes must be positive.")

    sine_matrix, cosine_matrix = fourier_parameter_matrices(layers, min(modes, layers))
    u, *_ = np.linalg.lstsq(sine_matrix, gamma_array, rcond=None)
    v, *_ = np.linalg.lstsq(cosine_matrix, beta_array, rcond=None)
    return u, v


def fourier_to_angles(
    layers: int,
    u: np.ndarray | list[float],
    v: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    u_array = np.asarray(u, dtype=float)
    v_array = np.asarray(v, dtype=float)
    if u_array.ndim != 1 or v_array.ndim != 1:
        raise ValueError("u and v must be one-dimensional arrays.")
    if len(u_array) != len(v_array):
        raise ValueError("u and v must have the same length.")
    if len(u_array) == 0:
        raise ValueError("At least one Fourier mode is required.")

    sine_matrix, cosine_matrix = fourier_parameter_matrices(layers, len(u_array))
    gamma = sine_matrix @ u_array
    beta = cosine_matrix @ v_array
    return gamma, beta


def _wrap_qaoa_angles(gamma: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wrapped_gamma = np.mod(gamma, np.pi)
    wrapped_beta = np.mod(beta, np.pi)
    wrapped_beta = np.where(wrapped_beta > np.pi / 2, np.pi - wrapped_beta, wrapped_beta)
    return wrapped_gamma, wrapped_beta


def canonicalize_qaoa_params(params: np.ndarray | list[float]) -> np.ndarray:
    params_array = np.asarray(params, dtype=float)
    if params_array.ndim != 1 or len(params_array) % 2 != 0:
        raise ValueError("params must contain concatenated gamma and beta values.")

    layers = len(params_array) // 2
    gamma = params_array[:layers]
    beta = params_array[layers:]
    wrapped_gamma, wrapped_beta = _wrap_qaoa_angles(gamma, beta)
    return np.concatenate([wrapped_gamma, wrapped_beta])


def fourier_extrapolated_initial_params(
    previous_params: np.ndarray | list[float],
    target_layers: int,
    modes: int | None = None,
) -> np.ndarray:
    previous_array = np.asarray(previous_params, dtype=float)
    if previous_array.ndim != 1 or len(previous_array) % 2 != 0:
        raise ValueError("previous_params must contain concatenated gamma and beta values.")

    previous_layers = len(previous_array) // 2
    if target_layers <= 0:
        raise ValueError("target_layers must be positive.")

    gamma = previous_array[:previous_layers]
    beta = previous_array[previous_layers:]
    mode_count = min(previous_layers, target_layers, modes if modes is not None else previous_layers)
    u, v = angles_to_fourier(gamma, beta, modes=mode_count)
    extrapolated_gamma, extrapolated_beta = fourier_to_angles(target_layers, u, v)
    wrapped_gamma, wrapped_beta = _wrap_qaoa_angles(extrapolated_gamma, extrapolated_beta)
    return np.concatenate([wrapped_gamma, wrapped_beta])


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
    simulator_backend=None,
) -> dict[str, int]:
    canonical_params = canonicalize_qaoa_params(params)

    if simulator_backend is not None:
        bound_circuit = circuit.assign_parameters(canonical_params)
        compiled_circuit = transpile(bound_circuit, simulator_backend, optimization_level=0, seed_transpiler=seed)
        result = simulator_backend.run(compiled_circuit, shots=shots).result()
        return result.get_counts(0)

    sampler = StatevectorSampler(default_shots=shots, seed=seed)
    return sampler.run([(circuit, canonical_params)]).result()[0].data.meas.get_counts()


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


def _random_initial_params(layers: int, rng: np.random.Generator) -> np.ndarray:
    return np.concatenate(
        [
            rng.uniform(0.0, np.pi, size=layers),
            rng.uniform(0.0, np.pi / 2, size=layers),
        ]
    )


def _finite_difference_gradient(
    objective,
    params: np.ndarray,
    step_size: float,
) -> np.ndarray:
    if step_size <= 0.0:
        raise ValueError("gradient_step must be positive.")

    gradient = np.zeros_like(params, dtype=float)
    for index in range(len(params)):
        shift = np.zeros_like(params, dtype=float)
        shift[index] = step_size
        forward = objective(canonicalize_qaoa_params(params + shift))
        backward = objective(canonicalize_qaoa_params(params - shift))
        gradient[index] = (forward - backward) / (2.0 * step_size)
    return gradient


def optimize_qaoa(
    graph: nx.Graph,
    layers: int,
    shots: int,
    restarts: int,
    seed: int,
    initial_params: np.ndarray | list[float] | None = None,
    optimizer: str = "COBYLA",
    sgd_config: SGDConfig | None = None,
    simulator_backend=None,
) -> OptimizationResult:
    circuit, _, _ = build_qaoa_circuit(graph, layers)
    rng = np.random.default_rng(seed)
    iteration_history: list[float] = []
    optimizer_name = optimizer.lower()

    if sgd_config is None:
        sgd_config = SGDConfig()

    def expected_cut_value(params: np.ndarray) -> float:
        counts = sample_counts(
            circuit,
            params,
            shots=shots,
            seed=seed,
            simulator_backend=simulator_backend,
        )
        expected_cut, _ = evaluate_counts(counts, graph)
        return expected_cut

    def objective(params: np.ndarray) -> float:
        return -expected_cut_value(params)

    initial_guesses: list[np.ndarray] = []

    if initial_params is not None:
        warm_start = canonicalize_qaoa_params(initial_params)
        if warm_start.shape != (2 * layers,):
            raise ValueError(f"Expected {2 * layers} initial parameters, received {warm_start.shape}.")
        initial_guesses.append(warm_start)

    for _ in range(max(restarts - len(initial_guesses), 0)):
        initial_guesses.append(_random_initial_params(layers, rng))

    if not initial_guesses:
        initial_guesses.append(_random_initial_params(layers, rng))

    best_x = None
    best_fun = None

    if optimizer_name == "cobyla":
        bounds = [(0.0, np.pi)] * layers + [(0.0, np.pi / 2)] * layers
        for initial in initial_guesses:
            local_iteration_history: list[float] = []

            def tracked_objective(params: np.ndarray) -> float:
                expected_cut = expected_cut_value(params)
                local_iteration_history.append(expected_cut)
                return -expected_cut

            result = minimize(tracked_objective, np.asarray(initial, dtype=float), method="COBYLA", bounds=bounds)
            if best_fun is None or result.fun < best_fun:
                best_fun = float(result.fun)
                best_x = np.asarray(result.x, dtype=float)
                iteration_history = local_iteration_history.copy()
    elif optimizer_name == "sgd":
        for initial in initial_guesses:
            params = canonicalize_qaoa_params(initial)
            current_expected_cut = expected_cut_value(params)
            current_value = -current_expected_cut
            local_best_x = params.copy()
            local_best_fun = current_value
            local_iteration_history = [current_expected_cut]

            for step in range(sgd_config.steps):
                gradient = _finite_difference_gradient(objective, params, sgd_config.gradient_step)
                learning_rate = sgd_config.learning_rate / (1.0 + sgd_config.learning_rate_decay * step)
                params = canonicalize_qaoa_params(params - learning_rate * gradient)
                current_expected_cut = expected_cut_value(params)
                current_value = -current_expected_cut
                local_iteration_history.append(current_expected_cut)
                if current_value < local_best_fun:
                    local_best_fun = current_value
                    local_best_x = params.copy()

            if best_fun is None or local_best_fun < best_fun:
                best_fun = local_best_fun
                best_x = local_best_x.copy()
                iteration_history = local_iteration_history.copy()
    else:
        raise ValueError(f"Unsupported optimizer {optimizer!r}. Use 'COBYLA' or 'SGD'.")

    if best_x is None or best_fun is None:
        raise RuntimeError("Optimization did not produce a result.")

    canonical_best = canonicalize_qaoa_params(best_x)
    final_counts = sample_counts(
        circuit,
        canonical_best,
        shots=shots,
        seed=seed,
        simulator_backend=simulator_backend,
    )
    expected_cut, cut_distribution = evaluate_counts(final_counts, graph)
    return OptimizationResult(
        canonical_best,
        expected_cut,
        final_counts,
        cut_distribution,
        iteration_history,
    )


def solve_maxcut_instance(
    edges: Iterable[tuple[int, int]] | None = None,
    *,
    graph: nx.Graph | None = None,
    layers: int = 2,
    shots: int = 2048,
    restarts: int = 8,
    seed: int = 7,
    initial_params: np.ndarray | list[float] | None = None,
    optimizer: str = "COBYLA",
    sgd_config: SGDConfig | None = None,
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
        initial_params=initial_params,
        optimizer=optimizer,
        sgd_config=sgd_config,
        simulator_backend=simulator_backend,
    )
    exact_cut, exact_assignments = brute_force_maxcut(graph)
    return MaxCutSolution(graph, result, exact_cut, exact_assignments)


def sweep_qaoa_layers(
    edges: Iterable[tuple[int, int]] | None = None,
    *,
    graph: nx.Graph | None = None,
    max_layers: int,
    shots: int = 2048,
    restarts: int = 8,
    seed: int = 7,
    fourier_modes: int | None = None,
    optimizer: str = "COBYLA",
    sgd_config: SGDConfig | None = None,
    simulator_backend=None,
) -> LayerSweepResult:
    if max_layers <= 0:
        raise ValueError("max_layers must be positive.")
    if fourier_modes is not None and fourier_modes <= 0:
        raise ValueError("fourier_modes must be positive when provided.")

    if graph is None:
        if edges is None:
            raise ValueError("Provide either edges or graph.")
        graph = build_graph(edges)

    exact_cut, exact_assignments = brute_force_maxcut(graph)
    entries: list[LayerSweepEntry] = []
    previous_params: np.ndarray | None = None

    for layers in range(1, max_layers + 1):
        initial_params = None
        if previous_params is not None:
            mode_count = min(fourier_modes if fourier_modes is not None else layers - 1, layers - 1)
            initial_params = fourier_extrapolated_initial_params(
                previous_params,
                target_layers=layers,
                modes=mode_count,
            )

        solution = solve_maxcut_instance(
            graph=graph,
            layers=layers,
            shots=shots,
            restarts=restarts,
            seed=seed,
            initial_params=initial_params,
            optimizer=optimizer,
            sgd_config=sgd_config,
            simulator_backend=simulator_backend,
        )
        optimal_probability = optimal_solution_probability(
            solution.result.counts,
            graph,
            exact_assignments,
        )
        entries.append(
            LayerSweepEntry(
                layers=layers,
                initial_params=initial_params,
                result=solution.result,
                optimal_probability=optimal_probability,
            )
        )
        previous_params = solution.result.x

    return LayerSweepResult(graph, exact_cut, exact_assignments, entries)


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
    sgd_config = SGDConfig(
        steps=args.sgd_steps,
        learning_rate=args.sgd_learning_rate,
        gradient_step=args.sgd_gradient_step,
        learning_rate_decay=args.sgd_learning_rate_decay,
    )
    solution = solve_maxcut_instance(
        edges=parse_edges(args.edges),
        layers=args.layers,
        shots=args.shots,
        restarts=args.restarts,
        seed=args.seed,
        optimizer=args.optimizer,
        sgd_config=sgd_config,
    )
    print_report(solution.graph, solution.result, args.layers)


if __name__ == "__main__":
    main()
