import glob
import os
import shutil
import matplotlib
from pinn_app import create_app, init_logger, redirect_std_streams

matplotlib.use('Agg')
logger = init_logger()
redirect_std_streams(logger)
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
for path in glob.glob(os.path.join(DATA_DIR, "*")):
    if os.path.isdir(path):
        try:
            shutil.rmtree(path)
        except OSError as e:
            pass

app = create_app()
server = app.server
