from sys import argv
from pathlib import Path
import json
import warnings

from causing import utils, estimate
from causing.examples import models
from causing.utils import print_output, round_sig_recursive, dump_json
from causing.graph import create_graphs, create_json_graphs, create_estimate_graphs
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
estimate_dat = estimate.estimate_models(m, xdat, mean_theo, estimate_input)
indiv_dat = create_indiv(m, xdat, show_nr_indiv)
graphs = create_json_graphs(m, xdat, indiv_dat, mean_theo, show_nr_indiv)

# Print text log
output_dir = Path("output") / model_name
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / "logging.txt", "w") as f:
    print_output(m, xdat, estimate_dat, estimate_input, indiv_dat, mean_theo, f)

# Print json output
dump_json(round_sig_recursive(graphs, 6), output_dir / "graphs.json")
estimate_dat = estimate.filter_important_keys(estimate_dat)
dump_json(round_sig_recursive(estimate_dat, 6), output_dir / "estimate.json")

# Draw graphs
create_graphs(m, graphs, output_dir, {}, show_nr_indiv)
create_estimate_graphs(m, estimate_dat, graphs, output_dir)
