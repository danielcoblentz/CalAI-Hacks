# convert_notebook.py

import os
import nbformat
from nbconvert import ScriptExporter

NOTEBOOK = "2Dto3D.ipynb"
OUTPUT_SCRIPT = "mesh_pipeline.py"

def main():
    # 1) sanity check
    if not os.path.exists(NOTEBOOK):
        print(f"ERROR: Notebook '{NOTEBOOK}' not found in {os.getcwd()}")
        return

    # 2) Read the notebook
    nb = nbformat.read(NOTEBOOK, as_version=4)

    # 3) Export to a single .py script
    exporter = ScriptExporter()
    script_body, _ = exporter.from_notebook_node(nb)

    # 4) Filter out pip installs if you like
    lines = script_body.splitlines()
    filtered = [ln for ln in lines if not ln.strip().startswith("pip install")]

    # 5) Write to mesh_pipeline.py
    with open(OUTPUT_SCRIPT, "w", encoding="utf-8") as f:
        f.write("# Auto-generated from 2Dto3D.ipynb\n")
        f.write("\n".join(filtered))
    print(f"Wrote merged script to '{OUTPUT_SCRIPT}'")

if __name__ == "__main__":
    main()