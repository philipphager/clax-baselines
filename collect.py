from pathlib import Path

import pandas as pd


def main():
    result_dir = Path("results")
    dfs = []

    for file in result_dir.glob("*test.csv"):
       df = pd.read_csv(file)
       dfs.append(df)

    if len(dfs) > 0:
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(result_dir / "results.csv", index=False)


if __name__ == '__main__':
    main()
