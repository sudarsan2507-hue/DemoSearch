with open(r"d:\Projects\DemoSearch\fags\failure_search.py", "r") as f:
    content = f.read()

# Update signature
old_sig = """    enable_re_verification: bool = True,
    shield_depth: int = 0,
) -> SearchResult:"""
new_sig = """    enable_re_verification: bool = True,
    shield_depth: int = 0,
    use_certificate: bool = True,
    certificate_bonus: float = 0.10,
) -> SearchResult:"""
content = content.replace(old_sig, new_sig)

# Update bonus logic
old_bonus = """                # Certificate Bonus
                if active_certificate is not None:
                    coherence = RELATION_COHERENCE.get((edge.relation, active_certificate), 0.3)
                    if coherence > 0.5:
                        s = min(1.0, s + 0.10)"""
new_bonus = """                # Certificate Bonus
                if use_certificate and active_certificate is not None:
                    coherence = RELATION_COHERENCE.get((edge.relation, active_certificate), 0.3)
                    if coherence > 0.5:
                        s = min(1.0, s + certificate_bonus)"""
content = content.replace(old_bonus, new_bonus)

with open(r"d:\Projects\DemoSearch\fags\failure_search.py", "w") as f:
    f.write(content)
print("patched failure_search.py parameters")
