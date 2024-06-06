import os
from datetime import datetime


def get_filenames(dir_path):
    for file_tuple in os.walk(dir_path):
        for filename in file_tuple[2]:
            try:
                dt = datetime.strptime(filename[19:27], "%Y%m%d")
                new_filename = dt.strftime("%Y_%m_%d.txt")
                os.rename(os.path.join(dir_path, filename), os.path.join(dir_path, new_filename))
            except Exception:
                pass


get_filenames("./Swarm")