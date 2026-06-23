with open(r"d:\Projects\DemoSearch\fags\failure_search.py", "r") as f:
    content = f.read()

# 1. Initialize is_post_revival
content = content.replace(
    "    active_certificate = None\n    current_revival_hops = 0",
    "    active_certificate = None\n    current_revival_hops = 0\n    is_post_revival = False"
)

# 2. Update the 4 recovery blocks
old_recovery_block = """            if current_revival_hops > 0:
                hops_survived_post_revival.append(current_revival_hops)
            current_revival_hops = 0
            if revived_rel is not None:
                active_certificate = revived_rel
                shield_hops_remaining = shield_depth"""

new_recovery_block = """            if current_revival_hops > 0:
                hops_survived_post_revival.append(current_revival_hops)
            current_revival_hops = 0
            if revived_rel is not None:
                active_certificate = revived_rel
                shield_hops_remaining = shield_depth
                is_post_revival = True
            else:
                is_post_revival = False"""

content = content.replace(old_recovery_block, new_recovery_block)

# 3. Update the traverse block
old_traverse = """        if active_certificate is not None:
            current_revival_hops += 1
            shield_hops_remaining = max(0, shield_hops_remaining - 1)
            if shield_hops_remaining == 0:
                active_certificate = None"""

new_traverse = """        if active_certificate is not None:
            shield_hops_remaining = max(0, shield_hops_remaining - 1)
            if shield_hops_remaining == 0:
                active_certificate = None
                
        if is_post_revival:
            current_revival_hops += 1"""

content = content.replace(old_traverse, new_traverse)

with open(r"d:\Projects\DemoSearch\fags\failure_search.py", "w") as f:
    f.write(content)
print("patched")
