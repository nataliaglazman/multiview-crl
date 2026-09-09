import json
import ast

with open('eval/view_latents.ipynb', 'r') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        try:
            ast.parse(source)
        except Exception as e:
            print(f"Cell {idx} syntax error: {e}")
            
print("Done checking syntax.")

def check_string(s):
    lines = source.split('\n')
    for i, l in enumerate(lines):
        if s in l:
            print(f"Cell {idx}, line {i}: {l}")

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        # check specifically for invalid accesses to all_content_indices
        check_string("len(all_content_indices)")
        check_string("len(all_style_indices)")
        check_string("[:, all_content_indices]")
        check_string("[:, all_style_indices]")
