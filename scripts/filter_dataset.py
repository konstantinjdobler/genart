import argparse
import pandas as pd

# Filter the annotations file to only contain painting

parser = argparse.ArgumentParser("Parsing arguments")

parser.add_argument("--source", "-s", type=str, required=True)
parser.add_argument("--destination", "-d", type=str, required=True)


if __name__ == "__main__":
    config = parser.parse_args()
    df = pd.read_csv(config.source, "\t")
    print(len(df))
    paintings_only = df[df["Is painting"] == "yes"]
    print(len(paintings_only))
    paintings_only.to_csv(config.destination, "\t", index=False)
