import pandas as pd

df = pd.read_csv("external/bd_online.csv")
pn = df[df["Pneumonia"] == 1]
nm = df[df["No Finding"] == 1]

has = pn["image_id"].str.contains("person", case=False)
print("pneumonia images: %d" % len(pn))
print("  with 'person' in the name: %d  (%.1f%%)" % (has.sum(), 100 * has.mean()))
print("  distinct person ids: %d"
      % pn.loc[has, "image_id"].str.extract(r"(person\d+)", expand=False).nunique())
print("  bacteria: %d   virus: %d"
      % (pn["image_id"].str.contains("bacteria", case=False).sum(),
         pn["image_id"].str.contains("virus", case=False).sum()))
print("normal images: %d" % len(nm))
print("  with 'person' in the name: %d"
      % nm["image_id"].str.contains("person", case=False).sum())