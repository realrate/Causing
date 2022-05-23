from sys import argv
from pathlib import Path
import warnings
import logging

import pandas

import causing
from causing import estimate
from causing.examples import models
from causing.utils import print_output, print_bias, round_sig_recursive, dump_json
from causing.graph import create_graphs, create_json_graphs, create_estimate_graphs
from causing.indiv import create_indiv

logging.basicConfig(level=logging.INFO)

# Our examples should run without any warnings, so let's treat them as errors.
warnings.filterwarnings("error")

# Keep wide output even if redirecting to file
pandas.set_option("display.max_columns", 500)
pandas.set_option("display.max_rows", 500)
pandas.set_option("display.width", 500)

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
estimate_dat = causing.estimate_models(m, xdat, mean_theo, estimate_input)
biases, biases_std = causing.estimate_biases(
    m, xdat, estimate_input["ymvars"], estimate_input["ymdat"]
)
indiv_dat = create_indiv(m, xdat, show_nr_indiv)
graphs = create_json_graphs(m, xdat, indiv_dat, mean_theo, show_nr_indiv)

# Print text log
output_dir = Path("output") / model_name
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / "logging.txt", "w") as f:
    print_output(m, xdat, estimate_dat, estimate_input, indiv_dat, mean_theo, f)
    print_bias(m, biases, biases_std, f)

# Print json output
dump_json(round_sig_recursive(graphs, 6), output_dir / "graphs.json")
estimate_dat = estimate.filter_important_keys(estimate_dat)
dump_json(round_sig_recursive(estimate_dat, 6), output_dir / "estimate.json")

# Draw graphs
create_graphs(m, graphs, output_dir, {}, None, show_nr_indiv=show_nr_indiv)
create_estimate_graphs(m, estimate_dat, graphs, output_dir)
