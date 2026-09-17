import json
import ast

with open('eval/view_latents.ipynb', 'r') as f:
    nb = json.load(f)

print("Starting AST syntax check for all code cells...")
error_count = 0

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        try:
            # ast.parse validates that the Python code has valid syntax
            ast.parse(source)
            print(f"  ✓ Cell {idx} passed syntax check.")
        except Exception as e:
            print(f"  ✗ Cell {idx} syntax error: {e}")
            error_count += 1
            
if error_count == 0:
    print("SUCCESS: All code cells in the notebook have valid Python syntax!")
else:
    print(f"FAILED: Found {error_count} syntax errors.")
