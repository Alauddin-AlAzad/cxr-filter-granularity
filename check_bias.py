"""
Cross-tabulates image size against label.

If the two classes came from different sources, the size distribution gives it
away, and any score on this set would partly measure provenance rather than
pathology.

    python check_bias.py external/bd_online.csv
"""

import sys

import pandas as pd


def bucket(n):
    if n < 400:
        return "under 400"
    if n < 800:
        return "400-800"
    if n < 1500:
        return "800-1500"
    return "over 1500"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "external/bd_online.csv"
    df = pd.read_csv(path)
    df["label"] = df["Pneumonia"].map({1: "pneumonia", 0: "normal"})
    df["long_side"] = df[["width", "height"]].max(axis=1)
    df["size_group"] = df["long_side"].map(bucket)

    print("exact size by label:")
    top = df.groupby(["width", "height"]).size().sort_values(ascending=False).head(8).index
    sub = df.set_index(["width", "height"]).loc[list(top)].reset_index()
    print(pd.crosstab([sub.width, sub.height], sub.label).to_string())

    print("\nsize group by label:")
    print(pd.crosstab(df.size_group, df.label).to_string())

    print("\ncolour mode by label:")
    print(pd.crosstab(df["mode"], df.label).to_string())

    print("\nfile name prefix by label:")
    df["prefix"] = df.image_id.str.replace(r"[-_]?\d+$", "", regex=True).str.lower().str[:20]
    print(pd.crosstab(df.prefix, df.label).head(12).to_string())


if __name__ == "__main__":
    main()