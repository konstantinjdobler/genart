import os
import pandas as pd
import argparse

# steps to preprocess dataset script
# 1 get all painting names of artemis dataset
# 2 compare with downloaded images
# exchange - with _
# 3 save existing pictures in csv with -
# 4 go through files by name
# exchange - with _
# 5 save existing pictures in directory with -
# can be done with get filenames script

parser = argparse.ArgumentParser("Parsing: ")

parser.add_argument('--data-dir', '-d', type=str, default=".",
                    help="Specifiy the directory name with files")
parser.add_argument('--file', '-f', type=str, default="downloaded_files.txt",
                    help="Specifiy the file name for input file")
parser.add_argument('--split', '-s', type=str, default="_",
                    help="Specifiy split symbol")

config = parser.parse_args()


def rename_files_in_directory(csv_file="artEmis_dataset", src_dir=".", concat_str="_"):
    pd_csv = pd.read_csv(csv_file)
    unique_elements = pd_csv["painting"].apply(
        lambda s:  concat_str.join(s.split("/")[-2:]) + ".jpg").unique()

    for name in unique_elements:
        try:
            cur_name = name.replace(concat_str, "-")

            os.rename(cur_name, name)
        except:
            print(cur_name + " not found")


rename_files_in_directory(config.file, config.data_dir, config.split)
