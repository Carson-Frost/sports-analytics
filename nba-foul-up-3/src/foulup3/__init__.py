"""Should an NBA team up 3 foul or defend? A play-by-play model of the final seconds."""

FIRST, LAST = 1996, 2025  # seasons 1996-97 to 2025-26, named by the year they start
SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(FIRST, LAST + 1)]
# the model uses the high-foul era on the trend chart; older seasons only show the trend
MODEL_SEASONS = SEASONS[-5:]
