"""Build API graphs; canvas JSON is exported by the real ComfyUI frontend."""
import importlib.util
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("relay_workflow_recipes", root / "h3_relay/workflows.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
destination = root / "example_workflows/toolkit"
destination.mkdir(exist_ok=True)
for name, graph in (("H3-Character-Swap", module.character_swap_graph()),
                    ("H3-Person-Removal", module.person_removal_graph())):
    path = destination / (name + ".api.json")
    path.write_text(json.dumps(graph, indent=2) + "\n")
    print(path)
