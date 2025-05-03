import glob
import os
from pinn_app import create_app, init_logger, redirect_std_streams

if __name__ == "__main__":
    logger = init_logger()
    redirect_std_streams(logger)
    for f in glob.glob(os.path.join("..", "data", "*.npz")):
        try:
            os.remove(f)
        except OSError:
            pass
    app = create_app()
    app.run(
        host="0.0.0.0",
        port=8050,
        debug=True,
        use_reloader=False
    )
