import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import main

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data")
    main(p.parse_args().out_dir)
