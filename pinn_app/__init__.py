from .layout import create_layout
from .callbacks import register_callbacks
from .logger import init_logger, redirect_std_streams


def create_app():
    """
    Create and configure the Dash application.
    This factory function builds the app layout and registers all callbacks.
    :return: A Dash app instance with layout and callbacks applied
    """
    app = create_layout()
    register_callbacks(app)
    return app


__all__ = [
    "create_app",
    "init_logger",
    "redirect_std_streams",
]
