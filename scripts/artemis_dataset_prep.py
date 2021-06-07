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
parser.add_argument('--concat', '-c', type=str, default="_",
                    help="Specifiy concat string")
parser.add_argument('--old-string', '-o', type=str, default="_",
                    help="Specifiy old string in current files that should be replaced")
parser.add_argument('--extension', '-e', type=str, default=".jpg",
                    help="Specifiy file extension")

config = parser.parse_args()
print(config)

# TODO Make renaming more generic with more parsing arguments


def rename_files_in_directory(csv_file="artEmis_dataset", src_dir=".", concat_str="_", old_string="-", extension=".jpg"):
    pd_csv = pd.read_csv(csv_file)
    print(pd_csv)
    unique_elements = pd_csv["painting"].apply(
        lambda s:  concat_str.join(s.split("/")[-2:]) + extension).unique()

    for name in unique_elements:
        try:
            # the names have to match the given file
            old_name = name.replace(concat_str, old_string)

            old = os.path.join(config.data_dir, old_name)
            new = os.path.join(config.data_dir, name)

            os.rename(old, new)
        except:
            print(old_name + " not found")


rename_files_in_directory(config.file, config.data_dir,
                          config.concat, config.old_string, config.extension)
