from pathlib import Path

from matplotlib import font_manager

from foulup3 import charts


def test_figures_draw_in_the_vendored_liberation_sans():
    for weight in ("normal", "bold"):
        props = font_manager.FontProperties(weight=weight)  # the family comes from rcParams
        path = Path(font_manager.findfont(props, fallback_to_default=False))
        assert path.parent == charts.FONTS
        assert font_manager.get_font(path).family_name == "Liberation Sans"
