import os
import pandas as pd
PATH = "/share/j_sun/MABe_dataset/MABe25_videos"
DEST_PATH = "/share/j_sun/MABe_dataset/MABe25_videos_splitted"
metadata = pd.read_csv("kaggle_metadata.csv")
def parse(file_name: str) -> str: 
    # name_<something>...<something>_number.mp4 -> name_number.mp4
    name = file_name.split("_")[0]
    number = file_name.split("_")[-1].split(".")[0]
    return f"{name}_{number}.mp4"

split = {parse(row.name): row.split for row in metadata.itertuples()}

for file in os.listdir(PATH):
    if file.endswith(".mp4"): 
        #extract the string before 
        if file in split:
            split_name = split[file]
            dest_folder = os.path.join(DEST_PATH, split_name)
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            os.rename(os.path.join(PATH, file), os.path.join(dest_folder, file))