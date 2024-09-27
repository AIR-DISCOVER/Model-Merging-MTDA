# Auto parse log and extract mIoU
import os
import re
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, required=True)
parser.add_argument("--csv_save_path", type=str, default="log_data_to_save.csv")
args = parser.parse_args()

dir = args.log_dir

csv_filename = args.csv_save_path

val_dataset = ' '
with open(csv_filename, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Log File Name', 'Val on','mIoU'])

    for filename in os.listdir(dir):
        if filename.endswith('.log'):
            if '-cs' in filename:
                val_dataset = "cs"
            elif '-idd' in filename:
                val_dataset = "idd"
            elif '-darkzurich' in filename:
                val_dataset = "darkzurich"
            elif '-acdc' in filename:
                val_dataset = "acdc"
                
                
            file_path = os.path.join(dir, filename)

            with open(file_path, 'r') as file:
                lines = file.readlines()

            line = lines[-2].strip()
            # print(line)
            match = re.findall(r'(\d+\.\d+)', line)
            # print(match)
            if match:
                mIoU = match[1]
                csv_writer.writerow([filename, val_dataset, mIoU])
            else:
                csv_writer.writerow([filename, val_dataset, 'Not Found'])
