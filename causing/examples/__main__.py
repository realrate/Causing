from sys import argv

from causing.causing import causing
from causing.examples import models

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
analyze_dat = causing(model_raw_dat)
