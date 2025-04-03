import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import time
from collections import deque

# Core Qiskit imports (pre-1.0)
from qiskit.utils import QuantumInstance
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA, SPSA, NELDER_MEAD

# Qiskit Optimization imports
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# For AerSimulator (local simulation)
from qiskit.providers.aer import AerSimulator

# ====================== CONFIGURATION ======================
CUSTOM_WEIGHTS = {
    (0, 1): 2.4,
    (0, 2): 2.8,
    (0, 4): 3.3,
    (1, 3): 3.0,
    (1, 4): 2.2,
    (4, 3): 2.0,
    (4, 2): 1.7,
    (2, 3): 1.7
}

# Node positions in 3D (x, y, z)
node_positions = {
        0: (0, 0, 0),
        1: (1, 2, 1),
        2: (2, 0, 2),
        3: (3, 1, 3),
        4: (1, 1, 3)
    }

# ====================== HELPER FUNCTIONS ======================
def create_graph():
    G = nx.DiGraph()
    
    for node, pos in node_positions.items():
        G.add_node(node, pos=pos)
    
    for (i, j), weight in CUSTOM_WEIGHTS.items():
        G.add_edge(i, j, weight=weight)
    
    return G, node_positions

def bfs_path(G, source, target):
    visited = set()
    queue = deque([[source]])
    
    while queue:
        path = queue.popleft()
        node = path[-1]
        
        if node == target:
            return path
        if node not in visited:
            visited.add(node)
            for neighbor in G.neighbors(node):
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
    return []

def dfs_path(G, source, target):
    visited = set()
    stack = [[source]]
    
    while stack:
        path = stack.pop()
        node = path[-1]
        
        if node == target:
            return path
        if node not in visited:
            visited.add(node)
            for neighbor in reversed(list(G.neighbors(node))):
                new_path = list(path)
                new_path.append(neighbor)
                stack.append(new_path)
    return []

def run_qaoa(G, source, target):
    # Formulate optimization problem
    qp = QuadraticProgram()
    
    for (i, j) in G.edges():
        qp.binary_var(f'x_{i}{j}')
    
    objective_linear = {f'x_{i}{j}': G[i][j]['weight'] for (i, j) in G.edges()}
    qp.minimize(linear=objective_linear)
    
    # Add flow constraints
    for node in G.nodes():
        out_edges = [f'x_{node}{j}' for j in G.successors(node)]
        in_edges = [f'x_{i}{node}' for i in G.predecessors(node)]
        coeffs = {**{var: 1 for var in out_edges}, **{var: -1 for var in in_edges}}
        rhs = 1 if node == source else (-1 if node == target else 0)
        qp.linear_constraint(coeffs, '==', rhs, f'flow_node_{node}')
    
    # Convert to QUBO
    qubo_converter = QuadraticProgramToQubo()
    qp_qubo = qubo_converter.convert(qp)
    
    # Run QAOA with optimized settings
    backend = AerSimulator(method="matrix_product_state")
    quantum_instance = QuantumInstance(backend, shots=512)  # Reduced shots for speed

    # qaoa = QAOA(optimizer=COBYLA(), reps=1, quantum_instance=quantum_instance)

    # # SPSA (good for noisy quantum simulations)
    qaoa = QAOA(optimizer=SPSA(maxiter=50), reps=2, quantum_instance=quantum_instance)

    # NELDER-MEAD (fewer iterations)
    # qaoa = QAOA(optimizer=NELDER_MEAD(maxfev=100), reps=2, quantum_instance=quantum_instance)

    optimizer = MinimumEigenOptimizer(qaoa)
    
    start_time = time.time()
    result = optimizer.solve(qp_qubo)
    qaoa_time = time.time() - start_time
    
    # Extract path
    path_edges = []
    for idx, var in enumerate(qp_qubo.variables):
        if result.x[idx] > 0.5:
            edge_name = var.name[2:]
            i, j = int(edge_name[0]), int(edge_name[1])
            path_edges.append((i, j))
    
    return path_edges, result.fval, qaoa_time

def run_benchmark(algorithms):
    results = {}
    G, _ = create_graph()
    source = 0
    target = 3
    
    for algo in algorithms:
        start_time = time.time()
        if algo == "Dijkstra":
            path = nx.dijkstra_path(G, source, target)
        elif algo == "A*":
            path = nx.astar_path(G, source, target, 
                               heuristic=lambda u, v: np.linalg.norm(np.array(G.nodes[u]['pos']) - np.array(G.nodes[v]['pos'])))
        elif algo == "Bellman-Ford":
            path = nx.bellman_ford_path(G, source, target)
        elif algo == "BFS":
            path = bfs_path(G, source, target)
        elif algo == "DFS":
            path = dfs_path(G, source, target)
        elif algo == "QAOA":
            path_edges, _, _ = run_qaoa(G, source, target)
            path = reconstruct_path(path_edges, source, target)
        else:
            continue
            
        exec_time = time.time() - start_time
        cost = sum(G[u][v]['weight'] for u,v in zip(path[:-1], path[1:]))
        results[algo] = (path, cost, exec_time)
    
    return results

def reconstruct_path(edges, source, target):
    path = [source]
    current = source
    
    while current != target:
        for (u, v) in edges:
            if u == current:
                path.append(v)
                current = v
                break
    return path

def plot_comparison(results, node_positions):
    fig = plt.figure(figsize=(20, 10))
    colors = ['green', 'blue', 'orange', 'purple', 'cyan', 'red']
    
    for idx, (algo, (path, cost, time)) in enumerate(results.items()):
        ax = fig.add_subplot(2, 3, idx+1, projection='3d')
        
        # Plot all nodes
        for node, pos in node_positions.items():
            ax.scatter(pos[0], pos[1], pos[2], c='black', s=50)
            ax.text(pos[0], pos[1], pos[2], f'{node}', size=12, color='k')
        
        # Plot all edges
        for (i, j) in CUSTOM_WEIGHTS.keys():
            start = node_positions[i]
            end = node_positions[j]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                    c='gray', alpha=0.2)
        
        # Plot path
        path_edges = list(zip(path[:-1], path[1:]))
        for i, j in path_edges:
            start = node_positions[i]
            end = node_positions[j]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                    c=colors[idx], linewidth=3, label=f'{algo} Path')
        
        ax.set_title(f"{algo}\nCost: {cost:.2f}, Time: {time:.4f}s", fontsize=12)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    
    plt.tight_layout()
    plt.show()

# ====================== MAIN EXECUTION ======================
if __name__ == "__main__":
    G, node_positions = create_graph()
    
    # Benchmark different algorithms
    algorithms = ["Dijkstra", "A*", "Bellman-Ford", "BFS", "DFS", "QAOA"]
    results = run_benchmark(algorithms)
    
    # Print results
    print("\nAlgorithm Comparison:")
    print(f"{'Algorithm':<15} {'Time (s)':<10} {'Total Cost':<10} {'Path'}")
    for algo, (path, cost, time) in results.items():
        print(f"{algo:<15} {time:<10.4f} {cost:<10.4f} {path}")
    
    # Visualize results
    plot_comparison(results, node_positions)