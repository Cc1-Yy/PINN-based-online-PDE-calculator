import re
import string
from dash import Dash
from dash.dependencies import Input, Output


def register_input_validation(app: Dash):
    """
    Register a callback on the Dash app to validate the user's equation input.

    :param app: The Dash application instance to attach the validation callback to.
    :return: None. This function registers the callback and does not return a value.
    """
    @app.callback(
        Output("input-equation", "invalid"),
        Input("input-equation", "value"),
        prevent_initial_call=True,
    )
    def on_equation_change(expr: str) -> bool:
        """
        Validate the equation expression entered by the user.

        :param expr: The raw string value from the "input-equation" component.
        :return: True if the expression is invalid (to mark the input as invalid); False if it is valid.
        """
        if not expr:
            return False

        s = re.sub(r"\s+", "", expr)
        if re.search(r"[^0-9a-z_+\-*/\.\(\)]", s):
            return True

        op_re = r"(?:\+|\-|\*{1,2}|/)"
        if re.search(rf"^(?:{op_re})|(?:{op_re})$", s):
            return True

        num_re = r"(?:\d+(?:\.\d*)?|\.\d+)"
        base_re = r"(?:x|y|u|r)"
        letters = string.ascii_lowercase
        u_re = rf"u_[{letters}]{{1,2}}"
        var_re = rf"(?:{base_re}|{u_re})"
        token_re = rf"(?:{num_re}|{var_re})"
        expr_re = rf"(?:{token_re}(?:{op_re}{token_re})*)"
        paren_re = rf"\({expr_re}\)"
        atom_re = rf"(?:{num_re}|{var_re}|{paren_re})"
        full_re = re.compile(rf"{atom_re}(?:{op_re}{atom_re})*\Z")

        if not full_re.fullmatch(s):
            return True

        return False
