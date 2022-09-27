from sys import argv
from pathlib import Path
import warnings
import logging

import pandas

import causing.graph
from causing.examples import models
from causing.utils import round_sig_recursive, dump_json
from causing import create_indiv

logging.basicConfig(level=logging.INFO)  # type: ignore

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
m, xdat = model_function()
graphs = create_indiv(m, xdat, show_nr_indiv)

# Print json output
output_dir = Path("output") / model_name
dump_json(round_sig_recursive(graphs, 6), output_dir / "graphs.json")

# Draw graphs
annotated_graphs = causing.graph.annotated_graphs(
    m, graphs, ids=[str(i) for i in range(show_nr_indiv)]
)
causing.graph.create_graphs(annotated_graphs, output_dir / "graphs")
