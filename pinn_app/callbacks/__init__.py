from .input_validation import register_input_validation
from .bd_groups import register_bd_groups
from .training import register_training_callbacks
from .result_graph import register_result_graph
from .set_session_id import register_session_id


def register_callbacks(app):
    """
    Register all the app’s callback groups in one place.

    :param app: The Dash application instance to which callbacks will be attached.
    :return: None. This function simply calls each registration helper in turn.
    """
    register_input_validation(app)
    register_bd_groups(app)
    register_training_callbacks(app)
    register_result_graph(app)
    register_session_id(app)
