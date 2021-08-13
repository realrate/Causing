from sys import argv
from pathlib import Path
import json
import warnings

from causing import utils, estimate
from causing.examples import models
from causing.utils import print_output, round_sig_recursive
from causing.graph import create_graphs, create_json_graphs, sym_to_str
from causing.indiv import create_indiv

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

show_nr_indiv = 3

# Do all calculations
m, xdat, ymdat, estimate_input = model_function()
mean_theo = m.theo(xdat.mean(axis=1))
indiv_theos = utils.make_individual_theos(m, xdat, show_nr_indiv)
indiv_dat = create_indiv(m, xdat, indiv_theos, show_nr_indiv)
estimate_dat = estimate.estimate_models(m, xdat, mean_theo, estimate_input)
graphs = create_json_graphs(m, xdat, estimate_dat, indiv_dat, mean_theo, show_nr_indiv)
# Round to 6 significant figures to make results stable even with minor floating point inaccuracies
graphs = round_sig_recursive(graphs, sig=6)

# Print text log
output_dir = Path("output") / model_name
output_dir.mkdir(parents=True, exist_ok=True)
print_output(
    m,
    xdat,
    estimate_dat,
    estimate_input,
    indiv_dat,
    mean_theo,
    output_dir,
)

# Print json output
with open(output_dir / "graphs.json", "w") as f:
    json.dump(graphs, f, sort_keys=True, indent=4)

# Draw graphs
graphs["xnodes"] = [str(var) for var in m.xvars]
graphs["ynodes"] = [str(var) for var in m.yvars]
graphs["is_all_graph"] = True
graphs["final_var_is_rat_var"] = False
create_graphs(graphs, output_dir, {}, show_nr_indiv)
