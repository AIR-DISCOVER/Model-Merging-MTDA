import glob 
import os
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    directory = "/DATA_EDS/chenht/HRDA/HRDA/data/gta/images/"
    file_names = [i.split(".")[0]for i in os.listdir(directory)]
    random.seed(42)
    name_train, name_test = train_test_split(file_names, test_size=0.1, random_state=42)
    for name in tqdm(name_test):
        os.symlink(f"/DATA_EDS/chenht/HRDA/HRDA/data/gta/images/{name}.png", f"/DATA_EDS/chenht/HRDA/HRDA/data/gta_val/images/{name}.png")
        os.symlink(f"/DATA_EDS/chenht/HRDA/HRDA/data/gta/labels/{name}_labelTrainIds.png", f"/DATA_EDS/chenht/HRDA/HRDA/data/gta_val/labels/{name}_labelTrainIds.png")
    for name in tqdm(name_train):
        os.symlink(f"/DATA_EDS/chenht/HRDA/HRDA/data/gta/images/{name}.png", f"/DATA_EDS/chenht/HRDA/HRDA/data/gta_train/images/{name}.png")
        os.symlink(f"/DATA_EDS/chenht/HRDA/HRDA/data/gta/labels/{name}_labelTrainIds.png", f"/DATA_EDS/chenht/HRDA/HRDA/data/gta_train/labels/{name}_labelTrainIds.png")
  


