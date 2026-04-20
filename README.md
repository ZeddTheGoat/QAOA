# QAOA Max-Cut with Qiskit

This project implements QAOA with `qiskit` to solve an unweighted Max-Cut problem on a graph.

## Install Dependencies

```bash
python3 -m pip install -r requirements.txt
```

The noiseless workflow uses Qiskit's built-in `StatevectorSampler`. If you also want to run the noisy experiment in the notebook, install both `qiskit-aer` and `qiskit-ibm-runtime`.

If you want to open the notebook, you will usually also want a local Jupyter environment:

```bash
python3 -m pip install notebook ipykernel
```

## Run the Example

```bash
python3 qaoa_maxcut.py
```

If you want a step-by-step tutorial, use [qaoa_maxcut_tutorial.ipynb](/common/home/zz1010/projects/QAOA/qaoa_maxcut_tutorial.ipynb:1).
The notebook does not hide the graph instance inside the code: the main entry point is the `example_edges` cell at the top, so you can change that cell and solve your own graph directly.

The tutorial is organized around four main steps:

- start with a graph example
- construct the Hamiltonian and QAOA circuit
- solve the QAOA optimization problem
- visualize the sampling distribution and best cut

After the main noiseless experiment, the notebook also reruns the same instance with a noise model derived from a fake IBM backend, so you can compare noiseless and noisy behavior side by side.

The results section explicitly reports the probability of sampling a globally optimal solution, which makes it easier to compare success rates across different settings.

A typical way to start the notebook is:

```bash
jupyter notebook qaoa_maxcut_tutorial.ipynb
```

The default example graph is:

```text
(0,1), (0,2), (1,2), (1,3), (2,4), (3,4)
```

The script prints:

- graph nodes and edges
- optimized QAOA parameters `gamma` and `beta`
- the highest-frequency sampled bitstrings
- the cut values of those bitstrings
- the exact optimum found by brute force for comparison

## Custom Graphs

You can pass a graph through `--edges`:

```bash
python3 qaoa_maxcut.py --edges "0-1,1-2,2-3,3-0,0-2"
```

If you are working from Python or from the notebook, the recommended high-level entry point is:

```python
from qaoa_maxcut import solve_maxcut_instance

solution = solve_maxcut_instance(
    edges=[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
    layers=2,
    shots=2048,
    restarts=8,
    seed=7,
)
```

## Common Parameters

```bash
python3 qaoa_maxcut.py --layers 3 --shots 4096 --restarts 12 --seed 42
```

- `--layers`: QAOA depth `p`
- `--shots`: number of shots used for each objective evaluation and the final sampling run
- `--restarts`: number of random restarts for the classical optimizer
- `--seed`: random seed

## Code Structure

- `build_qaoa_circuit`: construct the QAOA circuit for Max-Cut
- `build_maxcut_hamiltonian`: construct the Max-Cut cost Hamiltonian
- `build_fake_backend`: instantiate a fake IBM backend
- `build_fake_backend_simulator`: build an Aer noisy simulator from a fake IBM backend
- `optimal_solution_probability`: compute the total probability of sampling a globally optimal solution
- `optimize_qaoa`: run the sampler and classical optimizer to search for QAOA parameters
- `solve_maxcut_instance`: solve a given graph instance and return both the QAOA result and the exact optimum
- `brute_force_maxcut`: compute the exact optimum by exhaustive search for validation
- `draw_graph`: visualize the graph and partitioned cuts in the notebook
- `plot_sampling_distribution`: plot the probability distribution over all possible bitstrings in fixed order, which makes symmetry easier to see
- `rank_assignments`: summarize the highest-frequency sampled assignments for presentation
