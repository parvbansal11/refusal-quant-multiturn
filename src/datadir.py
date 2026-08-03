"""Single source of truth for where data lives.

The repository keeps generated artifacts under data/, grouped by kind, while the
scripts refer to them by bare filename (completions_mt_3b_fp16.csv and so on).
This module maps a bare name to its home by prefix, so no script needs to know
the directory layout, and moving a group is a one-line change here.

    from datadir import path, find
    load(path("judged_mt_3b_fp16.csv"))     # -> .../data/judged/judged_mt_...
    for f in find("completions_*.csv"): ...  # globs the right subdir

Writers use path() too, so a run started from anywhere lands its output in the
correct group.
"""
import glob as _glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
BUNDLES = os.path.join(ROOT, "bundles")
PAPER = os.path.join(ROOT, "paper")


def bundle(name):
    """Absolute path to a refusal-direction .pt, which live outside data/.

    bundles/ is gitignored, so a fresh clone does not contain it; create it here
    so a first write (refusal_direction.py, ablation.py) does not fail on a
    missing parent directory.
    """
    os.makedirs(BUNDLES, exist_ok=True)
    return name if os.path.isabs(name) else os.path.join(BUNDLES, os.path.basename(name))

# prefix -> subdirectory. Order matters: judged2_ must be tested before judged_,
# and the longer incontext_cres_/incontext_pair_ names still live with incontext.
_GROUPS = [
    ("completions_", "completions"),
    ("judged2_", "judged"),
    ("judged_", "judged"),
    ("incontext_", "incontext"),
    ("singleturn_", "singleturn"),
    ("validation_", "validation"),
]


def _group(name):
    base = os.path.basename(name)
    for prefix, sub in _GROUPS:
        if base.startswith(prefix):
            return sub
    return ""          # advbench_*, scenarios.json and friends sit at data/ root


def path(name):
    """Absolute location of a bare data filename."""
    if os.path.isabs(name):
        return name
    return os.path.join(DATA, _group(name), os.path.basename(name))


def find(pattern):
    """glob a bare pattern inside whichever group it belongs to."""
    return sorted(_glob.glob(os.path.join(DATA, _group(pattern), pattern)))
