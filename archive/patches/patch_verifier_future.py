def fix_verifier():
    with open(r"d:\Projects\DemoSearch\fags\verifier.py", "r", encoding="utf-8") as f:
        content = f.read()

    # Locate the __future__ import and place it at the very top
    # Let's remove any instances of it first, and insert it back at line 1.
    content = content.replace("from __future__ import annotations\n", "")
    content = content.replace("from __future__ import annotations", "")
    
    # Put it at the very top
    content = "from __future__ import annotations\n" + content
    
    with open(r"d:\Projects\DemoSearch\fags\verifier.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Fixed verifier.py __future__ import order")

if __name__ == "__main__":
    fix_verifier()
