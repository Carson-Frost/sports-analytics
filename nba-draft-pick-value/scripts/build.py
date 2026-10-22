"""Raw draft pages to data/draftees.csv, one row per draftee."""

from pathlib import Path

from draft_value.data import YEARS, read_raw

ROOT = Path(__file__).resolve().parents[1]

draftees = read_raw(ROOT / "data" / "raw")
missing = sorted(set(YEARS) - set(draftees["year"]))
if missing:
    raise SystemExit(f"Missing draft pages for {missing}; run scripts/pull.py")
draftees.to_csv(ROOT / "data" / "draftees.csv", index=False)
print(f"{len(draftees)} draftees, {draftees['year'].nunique()} drafts -> data/draftees.csv")
