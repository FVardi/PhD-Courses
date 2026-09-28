"""Fetch FordA, ArticularyWordRecognition, Epilepsy and FordB from the UCR/UEA archive into data/raw/.

Downloads and verifies, nothing more. Splitting, the stratified 10% subsets, scaling and
array construction belong to dev/1_prepare_data.py, so that "the official partition,
untouched" stays something you can check by looking at data/raw/.

FD-A and FD-B are deliberately NOT fetched. Part F reads them from inside the unchanged TF-C
checkout (see tfc_data in src/config.yaml); a separately obtained copy could be preprocessed
differently and would quietly invalidate the reproduction.

Safe to re-run: a dataset that already verifies is left alone, so this also serves to check a
hand-copied course-supplied data folder. Use --force to re-download regardless.

    python dev/0_fetch_data.py                 # required + optional datasets
    python dev/0_fetch_data.py --verify-only   # check what is on disk, download nothing
    python dev/0_fetch_data.py --datasets Epilepsy --force
"""

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "src" / "config.yaml"
ARCHIVE = "https://timeseriesclassification.com/aeon-toolkit/{name}.zip"
TIMEOUT = 600


def read_ts_header(path: Path) -> dict:
    """Read the @-header and count the data rows of a UCR/UEA .ts file.

    The .ts format is the only one the archive ships for every dataset: the univariate
    problems also carry _TRAIN.txt, but the multivariate ones do not.
    """
    info = {"channels": 1, "length": None, "classes": None, "missing": None, "n_series": 0}
    in_data = False
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if in_data:
                if line:
                    info["n_series"] += 1
                continue
            if not line or line.startswith("#"):
                continue
            low = line.lower()
            if low.startswith("@data"):
                in_data = True
            elif low.startswith("@dimensions"):
                info["channels"] = int(line.split()[1])
            elif low.startswith("@serieslength"):
                info["length"] = int(line.split()[1])
            elif low.startswith("@missing"):
                info["missing"] = line.split()[1].lower() == "true"
            elif low.startswith("@classlabel"):
                parts = line.split()
                info["classes"] = len(parts) - 2 if len(parts) > 2 else None
    return info


def verify(name: str, raw_dir: Path, expect: dict) -> list[str]:
    """Return a list of problems; an empty list means the dataset on disk is usable."""
    problems = []
    headers = {}
    for split, key in (("TRAIN", "n_train"), ("TEST", "n_test")):
        path = raw_dir / f"{name}_{split}.ts"
        if not path.exists():
            problems.append(f"missing {path.relative_to(ROOT)}")
            continue
        h = headers[split] = read_ts_header(path)
        if h["n_series"] != expect[key]:
            problems.append(
                f"{split}: {h['n_series']} series on disk, config says {expect[key]}"
            )
        for field in ("channels", "length", "classes"):
            if h[field] is not None and h[field] != expect[field]:
                problems.append(
                    f"{split}: {field} is {h[field]} on disk, config says {expect[field]}"
                )
        if h["missing"]:
            problems.append(f"{split}: header declares missing values")
    return problems


def download(name: str, raw_dir: Path) -> None:
    url = ARCHIVE.format(name=name)
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        blob = response.read()
    print(f"  {len(blob) / 1e6:.1f} MB, extracting")
    raw_dir.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(blob)).extractall(raw_dir)


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    known = cfg["datasets"]["required"] + cfg["datasets"].get("optional", [])

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", choices=known, default=known)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--verify-only", action="store_true", help="never download")
    args = parser.parse_args()

    raw_root = ROOT / cfg["paths"]["data_raw"]
    failed = []

    for name in args.datasets:
        expect = cfg["datasets"]["meta"][name]
        raw_dir = raw_root / name
        print(f"{name}")

        problems = verify(name, raw_dir, expect)
        if problems and not args.verify_only:
            if args.force or not raw_dir.exists():
                download(name, raw_dir)
            else:
                print("  present but does not verify, re-downloading")
                download(name, raw_dir)
            problems = verify(name, raw_dir, expect)
        elif not problems and args.force and not args.verify_only:
            download(name, raw_dir)
            problems = verify(name, raw_dir, expect)

        if problems:
            failed.append(name)
            for p in problems:
                print(f"  FAIL {p}")
        else:
            print(
                f"  ok  {expect['n_train']} train / {expect['n_test']} test, "
                f"{expect['channels']} ch x {expect['length']}, {expect['classes']} classes"
            )

    if failed:
        print()
        print(f"{len(failed)} dataset(s) unusable: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
