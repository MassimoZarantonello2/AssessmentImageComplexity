import csv
import os

csv_path = './transformation/files_csv/'
aj_csv_path = './transformation/aj_files_csv/'

csv_names = os.listdir(csv_path)

csv_dict = {}

for csv_name in csv_names:
    print(csv_name)
    with open(csv_path + csv_name,'r') as read_csv_file:
        csv_reader = csv.reader(read_csv_file, delimiter=',')
        values = next(csv_reader)
        for line in csv_reader:
            csv_dict[line[0]] = line[1:]
    csv_dict = dict(sorted(csv_dict.items(), key=lambda item: item[0]))
    
    with open(aj_csv_path +'ordered_'+ csv_name,'w',newline='') as write_csv_file:
        csv_writer = csv.writer(write_csv_file)
        csv_writer.writerow(values)
        for key in csv_dict:
            csv_writer.writerow([key] + csv_dict[key])
            