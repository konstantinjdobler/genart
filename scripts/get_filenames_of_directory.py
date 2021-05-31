import os
import argparse

parser = argparse.ArgumentParser("Parsing file and directory name")

parser.add_argument('--data_dir', '-d', type=str, default="./",
                    help="Specifiy the directory name with files")
parser.add_argument('--file', '-f', type=str, default="downloaded_files.txt",
                    help="Specifiy the file name for output file")
parser.add_argument('--constraint', '-c', type=str, default=".jpg",
                    help="Specifiy post fix for file names such as png/jpg")

config = parser.parse_args()
print(config)

write_file = config.file

f = open(write_file, "w")
f.close()

files = os.listdir(config.data_dir)
with open(write_file, "a") as output_file:
    for file in files:
        if file[-len(config.constraint):] == ".png" and file[0] != ".":
            output_file.write(f'{file}\n')
