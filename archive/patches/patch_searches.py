def patch():
    # 1. Patch baseline_search.py
    with open(r"d:\Projects\DemoSearch\fags\baseline_search.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    old_call = "s = verifier.score(query.keywords, edge, path_relations)"
    new_call = "s = verifier.score(query, edge, path_relations)"
    content = content.replace(old_call, new_call)
    
    with open(r"d:\Projects\DemoSearch\fags\baseline_search.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Patched baseline_search.py")

    # 2. Patch failure_search.py
    with open(r"d:\Projects\DemoSearch\fags\failure_search.py", "r", encoding="utf-8") as f:
        content = f.read()
        
    content = content.replace("s = verifier.score(query.keywords, edge, path_relations)", "s = verifier.score(query, edge, path_relations)")
    content = content.replace("query.keywords,", "query,")
    
    with open(r"d:\Projects\DemoSearch\fags\failure_search.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Patched failure_search.py")

if __name__ == "__main__":
    patch()
