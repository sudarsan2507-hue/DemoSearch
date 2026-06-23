with open(r"d:\Projects\DemoSearch\fags\failure_search.py", "r") as f:
    content = f.read()

old_code = """            if revived_rel is not None:
                if current_revival_hops > 0:
                    hops_survived_post_revival.append(current_revival_hops)
                current_revival_hops = 0
                active_certificate = revived_rel
                shield_hops_remaining = shield_depth"""

new_code = """            if current_revival_hops > 0:
                hops_survived_post_revival.append(current_revival_hops)
            current_revival_hops = 0
            if revived_rel is not None:
                active_certificate = revived_rel
                shield_hops_remaining = shield_depth"""

content = content.replace(old_code, new_code)

# Add logic for answer node finding
old_ans = """        # 6. Check answer node
        if current == query.answer_node:
            elapsed = time.perf_counter() - t0"""

new_ans = """        # 6. Check answer node
        if current == query.answer_node:
            if current_revival_hops > 0:
                hops_survived_post_revival.append(current_revival_hops)
            current_revival_hops = 0
            elapsed = time.perf_counter() - t0"""

content = content.replace(old_ans, new_ans)

with open(r"d:\Projects\DemoSearch\fags\failure_search.py", "w") as f:
    f.write(content)
print("fixed")
