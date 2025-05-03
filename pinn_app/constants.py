from collections import deque
from typing import Dict
import plotly.graph_objects as go

LOG_BUFFER: deque[str] = deque(maxlen=1000)
FIG_CACHE: Dict[str, go.Figure] = {}
FIG_PATHS = {
    "fig1": "../data/collocation_point_1.npz",
    "fig2": "../data/solution_residual_1.npz",
    "fig3": "../data/error_1.npz",
    "fig4": "../data/loss_1.npz",
    "fig5": "../data/boundary_loss_1.npz",
    "fig6": "../data/frequency_spectrum.npz",
    "fig7": "../data/collocation_point_2.npz",
    "fig8": "../data/solution_residual_2.npz",
    "fig9": "../data/error_2.npz",
    "fig10": "../data/loss_2.npz",
    "fig11": "../data/boundary_loss_2.npz",
}
