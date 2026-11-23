import importlib
import shutil
from pathlib import Path

from matplotlib import font_manager

from personnel import charts


def test_figures_draw_in_the_vendored_liberation_sans(tmp_path):
    fm, vendored = font_manager.fontManager, list(font_manager.fontManager.ttflist)
    for ttf in charts.FONTS.glob("*.ttf"):  # an installed copy, found first as on Linux runners
        fm.addfont(shutil.copy(ttf, tmp_path))
    fm.ttflist = [f for f in fm.ttflist if f not in vendored] + vendored
    importlib.reload(charts)
    for weight in ("normal", "bold"):
        props = font_manager.FontProperties(weight=weight)  # the family comes from rcParams
        path = Path(font_manager.findfont(props, fallback_to_default=False))
        assert path.parent == charts.FONTS
        assert font_manager.get_font(path).family_name == "Liberation Sans"
