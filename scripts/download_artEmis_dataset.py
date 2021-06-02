# Importing Necessary Modules
import requests  # to get image from the web
import shutil  # to save it locally
import csv
import pathlib
import os
from multiprocessing import Pool
import pandas as pd
from pathlib import Path


def _get_csv_entries(csv_file='artemis_dataset_release_v0.csv'):

    static_link = 'https://uploads7.wikiart.org/images/'

    pd_csv = pd.read_csv(csv_file)
    return pd_csv["painting"].apply(lambda s: static_link + s.replace("_", "/") + ".jpg").unique()


def _store_paintings_in_txt(link_list):
    with open("all_paintings.txt", "w+") as f:
        for url in link_list:
            f.write(f"{url}\n")


def prepare_next_run():
    # text_files_dict = {"a_p": "all_paintings.txt", "td_p": "to_download_paintings.txt", "d_p": "downloaded_paintings.txt", "nd_p": "not_downloaded_paintings.txt"}
    a_p = "all_paintings.txt"
    # TODO this can also be checked more efficiently with the get_filenames_of_directory.py repository after each run
    td_p = "to_download_paintings.txt"
    d_p = "downloaded_paintings.txt"
    nd_p = "not_downloaded_paintings.txt"

    with open(a_p, "r") as f:
        if len(f.readlines()) < 1:
            csv_entries = _get_csv_entries()
            _store_paintings_in_txt(csv_entries)

    crawled = []
    for file in [d_p, nd_p]:
        with open(file, "r") as f:
            crawled += f.readlines()

    with open(a_p, "r") as f:
        to_crawl = [x for x in f.readlines() if x not in set(crawled)]

    with open(td_p, "w+") as f:
        f.write("".join(to_crawl))

    return to_crawl


def _download_file(url):
    try:
        url = url.replace("\n", "")
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            # otherwise the downloaded image file's size will be zero
            r.raw.decode_content = True
        # "_".join("https://uploads7.wikiart.org/images/sam-francis/untitled-yellow-1961.jpg".split("/")[-2:]) == 'sam-francis-untitled-yellow-1961.jpg'
            # should be underscore '_' if taken from original artwork name
            filename = "_".join(url.split("/")[-2:])

            # define data storage for crawling run here
            external_directory = "/Volumes/Extreme SSD/artEmis_crawls/crawl6"
            path = os.path.join(external_directory, filename)
            with open(path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)

            # check file size and throw error if file size below 0!
            if Path(path).stat().st_size < 1:
                raise Exception("File size below 1 Byte")

            with open("downloaded_paintings.txt", "a") as f:
                f.write(f"{url}\n")

        else:
            print(f"{r.status_code} - {r.headers['Content-Type']} - {url}")

            with open("not_downloaded_paintings.txt", "a") as f:
                f.write(f"{url}\n")

    except Exception as e:
        print(
            f"Error: {str(e)} \n- by trying to download url (without 404): {url}")
        with open("not_downloaded_paintings.txt", "a") as f:
            f.write(f"{url}\n")


def download_in_parallel(link_list):
    with Pool(os.cpu_count()) as p:
        p.map(_download_file, link_list)


# test downloading of specific file
# _download_file('https://uploads7.wikiart.org/images/sam-francis/untitled-yellow-1961.jpg')
csv_list = prepare_next_run()
download_in_parallel(csv_list)
