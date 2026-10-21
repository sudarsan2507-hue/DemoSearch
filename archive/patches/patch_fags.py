import re

with open(r"d:\Projects\DemoSearch\fags\failure_search.py", "r") as f:
    content = f.read()

# 1. Update imports
content = content.replace(
    "from fags.verifier import Verifier",
    "from fags.verifier import Verifier, RELATION_COHERENCE"
)

# 2. Update failure_search signature
content = content.replace(
    "    enable_re_verification: bool = True,\n) -> SearchResult:",
    "    enable_re_verification: bool = True,\n    shield_depth: int = 0,\n) -> SearchResult:"
)

# 3. Initialize state variables
init_block = """    # Gold path tracking
    gold_path_pruned = False
    gold_path_recovered = False

    shield_hops_remaining = 0
    active_certificate = None
    current_revival_hops = 0
    hops_survived_post_revival = []"""
content = content.replace(
    "    # Gold path tracking\n    gold_path_pruned = False\n    gold_path_recovered = False",
    init_block
)

# 4. _handle_recovery call replacements
old_handle_recovery = """            current, path, path_relations, low_score_streak, failure_type, backtracks, recovery_attempts, recovery_successes, gold_path_recovered = _handle_recovery(
                graph, query, verifier, memory, path, path_relations, visited_nodes, visited_edges,
                local_alternatives, failed_relations, max_backtracks, enable_re_verification,
                backtracks, recovery_attempts, recovery_successes, gold_path_recovered, gold_nodes
            )
            if failure_type != FailureType.NONE:"""
            
new_handle_recovery = """            current, path, path_relations, low_score_streak, failure_type, backtracks, recovery_attempts, recovery_successes, gold_path_recovered, revived_rel = _handle_recovery(
                graph, query, verifier, memory, path, path_relations, visited_nodes, visited_edges,
                local_alternatives, failed_relations, max_backtracks, enable_re_verification,
                backtracks, recovery_attempts, recovery_successes, gold_path_recovered, gold_nodes
            )
            if revived_rel is not None:
                if current_revival_hops > 0:
                    hops_survived_post_revival.append(current_revival_hops)
                current_revival_hops = 0
                active_certificate = revived_rel
                shield_hops_remaining = shield_depth
            if failure_type != FailureType.NONE:"""
content = content.replace(old_handle_recovery, new_handle_recovery)

# 5. Candidate scoring (certificate bonus)
old_scoring = """            for edge in candidates:
                s = verifier.score(query.keywords, edge, path_relations)
                scored.append((s, edge))"""
new_scoring = """            for edge in candidates:
                s = verifier.score(query.keywords, edge, path_relations)
                
                # Certificate Bonus
                if active_certificate is not None:
                    coherence = RELATION_COHERENCE.get((edge.relation, active_certificate), 0.3)
                    if coherence > 0.5:
                        s = min(1.0, s + 0.10)
                        
                scored.append((s, edge))"""
content = content.replace(old_scoring, new_scoring)

# 6. Path misalignment check (relaxed floor)
old_misalignment = """        # 3. Path misalignment check
        if best_score < _RELEVANCE_FLOOR:"""
new_misalignment = """        # 3. Path misalignment check
        current_floor = 0.05 if shield_hops_remaining > 0 else _RELEVANCE_FLOOR
        if best_score < current_floor:"""
content = content.replace(old_misalignment, new_misalignment)

# 7. Traverse (shield tracking)
old_traverse = """        # Consume the candidate
        local_alternatives[depth].pop(0)
        
        current = best_edge.target"""
new_traverse = """        # Consume the candidate
        local_alternatives[depth].pop(0)
        
        if active_certificate is not None:
            current_revival_hops += 1
            shield_hops_remaining = max(0, shield_hops_remaining - 1)
            if shield_hops_remaining == 0:
                active_certificate = None
                
        current = best_edge.target"""
content = content.replace(old_traverse, new_traverse)

# 8. SearchResult additions
content = content.replace(
    "visited_node_set=visited_nodes,\n            )",
    "visited_node_set=visited_nodes,\n                hops_survived_post_revival=hops_survived_post_revival,\n            )"
)

# 9. _handle_recovery signature
content = content.replace(
    "    gold_nodes: list[str],\n) -> tuple[str, list[str], list[str], int, FailureType, int, int, int, bool]:",
    "    gold_nodes: list[str],\n) -> tuple[str, list[str], list[str], int, FailureType, int, int, int, bool, str | None]:"
)

# 10. _handle_recovery returns
content = content.replace(
    "return active_node, path, path_relations, 0, FailureType.NONE, backtracks, recovery_attempts, recovery_successes, gold_path_recovered",
    "return active_node, path, path_relations, 0, FailureType.NONE, backtracks, recovery_attempts, recovery_successes, gold_path_recovered, None"
)
content = content.replace(
    "return entry.target_id, new_path, new_relations, 0, FailureType.NONE, backtracks, recovery_attempts, recovery_successes, gold_path_recovered",
    "return entry.target_id, new_path, new_relations, 0, FailureType.NONE, backtracks, recovery_attempts, recovery_successes, gold_path_recovered, entry.relation"
)
content = content.replace(
    "return path[-1], path, path_relations, 0, FailureType.DEAD_END, backtracks, recovery_attempts, recovery_successes, gold_path_recovered",
    "return path[-1], path, path_relations, 0, FailureType.DEAD_END, backtracks, recovery_attempts, recovery_successes, gold_path_recovered, None"
)

with open(r"d:\Projects\DemoSearch\fags\failure_search.py", "w") as f:
    f.write(content)

print("failure_search.py patched successfully.")
