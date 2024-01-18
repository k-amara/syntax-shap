import matplotlib.pyplot as pl
import numpy as np
import pytest

from shap2.plots import colors
from shap2.plots._utils import convert_color


@pytest.mark.parametrize(
    ("color", "expected_result"),
    [
        ("shap_red", colors.red_rgb),
        ("shap_blue", colors.blue_rgb),
        ("null", "null"),
        ("jet", pl.get_cmap("jet")),
    ],
)
def test_convert_color(color, expected_result):
    result = convert_color(color)

    if isinstance(result, np.ndarray):
        assert np.allclose(result, expected_result)
    else:
        assert result == expected_result
