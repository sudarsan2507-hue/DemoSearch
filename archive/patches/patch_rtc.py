import re

def patch_file():
    with open(r"d:\Projects\DemoSearch\fags\failure_search.py", "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Update signature of failure_search
    sig_old = """    use_certificate: bool = True,
    certificate_bonus: float = 0.10,
) -> SearchResult:"""
    sig_new = """    use_certificate: bool = True,
    certificate_bonus: float = 0.10,
    use_rtc_lite: bool = False,
) -> SearchResult:"""
    content = content.replace(sig_old, sig_new)

    # 2. Add metrics variables
    vars_old = """    # Diagnostics
    edges_explored = 0
    backtracks = 0
    recovery_attempts = 0
    recovery_successes = 0"""
    vars_new = """    # Diagnostics
    edges_explored = 0
    backtracks = 0
    recovery_attempts = 0
    recovery_successes = 0
    
    trajectory_attempts = 0
    trajectory_matches = 0
    trajectory_utilities = 0
    matched_trajectories_in_this_run = 0"""
    content = content.replace(vars_old, vars_new)

    # 3. Update all _handle_recovery calls to pass use_rtc_lite
    call_old = """                local_alternatives, failed_relations, max_backtracks, enable_re_verification,
                backtracks, recovery_attempts, recovery_successes, gold_path_recovered, gold_nodes
            )"""
    call_new = """                local_alternatives, failed_relations, max_backtracks, enable_re_verification,
                backtracks, recovery_attempts, recovery_successes, gold_path_recovered, gold_nodes, use_rtc_lite
            )"""
    content = content.replace(call_old, call_new)

    # 4. Handle active_certificate assignment and trajectory matching
    cert_old = """            if revived_rel is not None:
                active_certificate = revived_rel
                shield_hops_remaining = shield_depth"""
    cert_new = """            if revived_rel is not None:
                active_certificate = revived_rel
                shield_hops_remaining = shield_depth
                
                if use_rtc_lite and isinstance(active_certificate, list) and len(active_certificate) > 1:
                    predicted_hop2_list = active_certificate[1]
                    check_depth = len(path) - 1
                    if check_depth < len(gold_relations):
                        trajectory_attempts += 1
                        if gold_relations[check_depth] in predicted_hop2_list:
                            trajectory_matches += 1
                            matched_trajectories_in_this_run += 1"""
    content = content.replace(cert_old, cert_new)

    # 5. Update candidate scoring with RTC-Lite boosts and veto
    scoring_old = """                # Certificate Bonus
                if use_certificate and active_certificate is not None:
                    coherence = RELATION_COHERENCE.get((edge.relation, active_certificate), 0.3)
                    if coherence > 0.5:
                        s = min(1.0, s + certificate_bonus)
                        
                scored.append((s, edge))"""
    scoring_new = """                # Certificate Bonus
                if use_rtc_lite and isinstance(active_certificate, list):
                    if current_revival_hops < len(active_certificate):
                        expected_for_hop = active_certificate[current_revival_hops]
                        if current_revival_hops == 0:
                            if edge.relation == expected_for_hop:
                                s = min(1.0, s + certificate_bonus)
                        elif current_revival_hops == 1:
                            if len(expected_for_hop) > 0 and edge.relation == expected_for_hop[0]:
                                s = min(1.0, s + certificate_bonus * 0.5)
                            elif len(expected_for_hop) > 1 and edge.relation == expected_for_hop[1]:
                                s = min(1.0, s + certificate_bonus * 0.25)
                elif use_certificate and active_certificate is not None and not isinstance(active_certificate, list):
                    coherence = RELATION_COHERENCE.get((edge.relation, active_certificate), 0.3)
                    if coherence > 0.5:
                        s = min(1.0, s + certificate_bonus)
                        
                # Verifier Veto
                if use_rtc_lite and s < 0.20:
                    s = 0.0
                        
                scored.append((s, edge))"""
    content = content.replace(scoring_old, scoring_new)

    # 6. Update answer node utility increment and SearchResult return (success=True)
    ans_old = """            if current_revival_hops > 0:
                hops_survived_post_revival.append(current_revival_hops)
            current_revival_hops = 0
            elapsed = time.perf_counter() - t0
            return SearchResult("""
    ans_new = """            if current_revival_hops > 0:
                hops_survived_post_revival.append(current_revival_hops)
            current_revival_hops = 0
            trajectory_utilities += matched_trajectories_in_this_run
            elapsed = time.perf_counter() - t0
            return SearchResult("""
    content = content.replace(ans_old, ans_new)

    ret1_old = """                edges_explored=edges_explored,
                visited_node_set=visited_nodes,
                hops_survived_post_revival=hops_survived_post_revival,
            )"""
    ret1_new = """                edges_explored=edges_explored,
                visited_node_set=visited_nodes,
                hops_survived_post_revival=hops_survived_post_revival,
                trajectory_attempts=trajectory_attempts,
                trajectory_matches=trajectory_matches,
                trajectory_utilities=trajectory_utilities,
            )"""
    content = content.replace(ret1_old, ret1_new)

    # 7. Update SearchResult return (success=False)
    ret2_old = """        visited_node_set=visited_nodes,
    )"""
    ret2_new = """        visited_node_set=visited_nodes,
        trajectory_attempts=trajectory_attempts,
        trajectory_matches=trajectory_matches,
        trajectory_utilities=trajectory_utilities,
    )"""
    content = content.replace(ret2_old, ret2_new)

    # 8. Update _handle_recovery signature
    h_sig_old = """    recovery_successes: int,
    gold_path_recovered: bool,
    gold_nodes: list[str],
) -> tuple[str, list[str], list[str], int, FailureType, int, int, int, bool, str | None]:"""
    h_sig_new = """    recovery_successes: int,
    gold_path_recovered: bool,
    gold_nodes: list[str],
    use_rtc_lite: bool = False,
) -> tuple[str, list[str], list[str], int, FailureType, int, int, int, bool, list | str | None]:"""
    content = content.replace(h_sig_old, h_sig_new)

    # 9. Update _handle_recovery trajectory generation
    gen_old = """        # Check if answer found immediately upon revival
        if entry.target_id == query.answer_node:
            recovery_successes += 1
        
        return entry.target_id, new_path, new_relations, 0, FailureType.NONE, backtracks, recovery_attempts, recovery_successes, gold_path_recovered, entry.relation"""
    gen_new = """        # Check if answer found immediately upon revival
        if entry.target_id == query.answer_node:
            recovery_successes += 1
            
        revived_rel = entry.relation
        if use_rtc_lite:
            expected = [revived_rel]
            neighbors = graph.get_neighbors(entry.target_id)
            scored_n = []
            for n_edge in neighbors:
                s = verifier.score(query.keywords, n_edge)
                scored_n.append((s, n_edge.relation))
            scored_n.sort(key=lambda x: x[0], reverse=True)
            
            if scored_n:
                hop2_predictions = [scored_n[0][1]]
                if len(scored_n) > 1:
                    hop2_predictions.append(scored_n[1][1])
                expected.append(hop2_predictions)
            revived_rel = expected
        
        return entry.target_id, new_path, new_relations, 0, FailureType.NONE, backtracks, recovery_attempts, recovery_successes, gold_path_recovered, revived_rel"""
    content = content.replace(gen_old, gen_new)
    
    with open(r"d:\Projects\DemoSearch\fags\failure_search.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    print("Patched successfully.")

if __name__ == "__main__":
    patch_file()
