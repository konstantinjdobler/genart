import os
import pandas as pd


if __name__ == "__main__":

    basepath = "./data/artEmis_128px"
    filepath = f"{basepath}/artemis_dataset_release_v0.csv"
    df = pd.read_csv(filepath)

    one_hot = pd.get_dummies(df["emotion"])
    df_emotion = df[["painting"]].join(one_hot)
    grouped = df_emotion.groupby("painting")
    
    # Check whether the emotion is existing
    result = grouped.max().reset_index()
    # print(result)

    # provisional
    def fix_name(s: str):
        x, _, y = s.partition("_")
        return f"{x}-{y}"
    
    result["painting"] = result["painting"].map(lambda x: f"{fix_name(x)}.jpg")
    print(len(result))

    def exists(x):
        file = f"{basepath}/processed2/{x}"
        return os.path.isfile(file)

    is_file = result["painting"].map(lambda x: exists(x))
    result = result[is_file]
    print(len(result))
    

    result.to_csv("./data/artEmis_128px/feature_dataset.csv", index=False)

    # def my_agg(series):
    #     return series.max()

    # print(t.agg(my_agg))
