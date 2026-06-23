import numpy as np
from fags.graph_generator import generate_dataset
from fags.verifier import Verifier
from fags.memory import create_memory
from fags.failure_search import failure_search

print("Running 100 queries with high noise...")
graph, queries = generate_dataset(num_nodes=200, num_queries=200, seed=42)
verifier = Verifier(noise_std=0.20, seed=42)

for shield in [0, 1, 2, 3]:
    memory = create_memory("top1", threshold=0.15)
    all_hops = []
    for q in queries:
        memory.clear()
        res = failure_search(graph, q, verifier, memory, shield_depth=shield)
        all_hops.extend(res.hops_survived_post_revival)
    if all_hops:
        print(f"Shield {shield} Mean: {np.mean(all_hops):.2f} (from {len(all_hops)} revivals)")
    else:
        print(f"Shield {shield} Mean: 0.00 (from 0 revivals)")
