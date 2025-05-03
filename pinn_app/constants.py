from collections import deque
from typing import Dict
import plotly.graph_objects as go

LOG_BUFFER: deque[str] = deque(maxlen=1000)
FIG_CACHE: Dict[str, go.Figure] = {}
