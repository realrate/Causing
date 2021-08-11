from sys import argv
from pathlib import Path
import json
import warnings

from causing import analyze
from causing.examples import models
from causing.utils import print_output, round_sig_recursive
from causing.graph import create_graphs, sym_to_str

# Our examples should run without any warnings, so let's treat them as errors.
warnings.filterwarnings("error")

if len(argv) != 2:
    print('Please call with model name as argument (e.g. "example" or "education").')
    exit(1)

model_name = argv[1]

try:
    model_function = getattr(models, model_name)
except AttributeError:
    print(f'Unkown model function "{model_name}".')
    exit(1)

model_raw_dat = model_function()
analyze_dat = analyze(model_raw_dat)
graphs = analyze_dat["graph_dat"]
# Round to 6 significant figures to make results stable even with minor floating point inaccuracies
graphs = round_sig_recursive(graphs, sig=6)

output_dir = Path("output") / model_name
output_dir.mkdir(parents=True, exist_ok=True)
print_output(
    analyze_dat["model_dat"],
    analyze_dat["estimate_dat"],
    analyze_dat["indiv_dat"],
    output_dir,
)
with open(output_dir / "graphs.json", "w") as f:
    json.dump(graphs, f, sort_keys=True, indent=4)

graphs["xnodes"] = [str(var) for var in analyze_dat["model_dat"]["xvars"]]
graphs["ynodes"] = [str(var) for var in analyze_dat["model_dat"]["yvars"]]
graphs["is_all_graph"] = True
graphs["final_var_is_rat_var"] = False

create_graphs(graphs, output_dir, {})
