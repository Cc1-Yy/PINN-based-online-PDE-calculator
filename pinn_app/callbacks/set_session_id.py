from dash.dependencies import Input, Output, State
import uuid


def register_session_id(app):
    """
    Add a callback that assigns a unique UUID to each browser session and
    stores it in session storage under 'session-id'.

    :param app: Dash app instance
    :return: None
    """
    @app.callback(
        Output('session-id', 'data'),
        Input('url', 'pathname'),
        State('session-id', 'data'),
        prevent_initial_call=False
    )
    def assign_session_id(pathname, sess):
        return sess or uuid.uuid4().hex
