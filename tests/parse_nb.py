import json, ast

with open('/home/ng24/projects/multiview-crl/eval/view_latents.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        code = "".join(cell['source'])
        try:
            ast.parse(code)
        except Exception as e:
            print(f"Cell {i} error: {e}")
