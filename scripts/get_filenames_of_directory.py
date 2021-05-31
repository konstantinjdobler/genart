import os
import argparse

parser = argparse.ArgumentParser("Parsing file and directory name")

parser.add_argument('--data_dir', '-d', type=str, default="./",
                    help="Specifiy the directory name with files")
parser.add_argument('--file', '-f', type=str, default="downloaded_files.txt",
                    help="Specifiy the file name for output file")
parser.add_argument('--prefix-constraint', '-pre', type=str, default=".",
                    help="Specifiy a prefix to skip file names starting with certain symbals such as . for hidden files")
parser.add_argument('--postfix-constraint', '-post', type=str, default=".jpg",
                    help="Specifiy a postfix to only match file names with a certain file extension such as .jpg")

config = parser.parse_args()
print(config)

write_file = config.file
prefix = config.prefix_constraint
postfix = config.postfix_constraint

f = open(write_file, "w")
f.close()

files = os.listdir(config.data_dir)
with open(write_file, "a") as output_file:
    for file in files:
        if file[-len(postfix):] == postfix and file[:len(prefix)] != prefix:
            output_file.write(f'{file}\n')
