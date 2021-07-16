from os.path import abspath, dirname
from lightfm_only import run_lightfm
from two_level_pipeline import run_two_level

model = "two_level" # "lfm" or "two_level"

n_comp = 50
epochs = 7

homedir = dirname(dirname(abspath(__file__)))
datadir = homedir + "/datasets/yelp/"

if model == "lfm":
    run_lightfm(datadir, n_comp, epochs)

if model == "two_level":
    run_two_level(n_comp, epochs, homedir, datadir)