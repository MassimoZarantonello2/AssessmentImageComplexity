import matplotlib.pyplot as plt
import csv
import os

def transformation_plots(save_path):
    transformation_path = './transformation/evaluated_results.csv'

    no_word = ['blur', 'brightness', 'contrast', 'hflip', 'hue','jpeg','noise','poosterize','saturation','MAE','RMSE','Pearsons','Spearmans','flipped']
    transformation_types = ['blur', 'brightness', 'contrast', 'hflip', 'hue','jpeg','noise','poosterize','saturation']

    transformation_list = [[],[],[],[],[]]

    with open(transformation_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row != []:                
                for i in range(len(row)):
                    if row[i] not in no_word:
                            transformation_list[i].append(float(row[i]))
                    else:
                        transformation_list[i].append(row[i])
    start_index = 1

    colors = ['r', 'g', 'b', 'y']
    for tras in transformation_types:
        if tras != 'hflip':
            end_index = start_index+5
            
            x_list = transformation_list[0][start_index:end_index]
            for i in range(1,5):
                y_list = transformation_list[i][start_index:end_index]
                plt.plot(x_list, y_list, color = colors[i-1], linewidth = 3, marker='o', markerfacecolor=colors[i-1], markersize=8,label = transformation_list[i][start_index-1])
                
            plt.title(tras)
            plt.xlabel('Params')
            plt.ylabel('Metrics')
            plt.legend(loc='upper right')
            plt.savefig(save_path+tras+'.png')
            plt.close()
            start_index += 6
        else:
            start_index += 2
            
def scatter_plots(save_path):
    csv_folder_path = 'transformation/aj_files_csv'
    csv_file_list = os.listdir(csv_folder_path)
    
    transformation_list = [[],[],[],[],[]]
    
    for csv_file in csv_file_list:
        csv_file_path = csv_folder_path + '/' + csv_file
        with open(csv_file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            first_row = next(reader)
            for row in reader:
                i = 0
                for param in row[1:]:
                    transformation_list[i].append(float(param))
                    i+=1
        for i in range(len(first_row)-1):
            x_list = [float(first_row[i+1]) for j in range(len(transformation_list[i]))]
            plt.scatter(x_list, transformation_list[i], label = csv_file)
        transformation_list = [[],[],[],[],[]]
        plt.title(first_row[0])
        plt.xlabel('Params')
        plt.ylabel('Complexity Score')
        plt.savefig(save_path+first_row[0]+'.png')
        plt.close()
        
def box_plot(save_path):
    csv_folder_path = 'transformation/aj_files_csv'
    csv_file_list = os.listdir(csv_folder_path)
    
    transformation_list = [[],[],[],[],[]]
    
    for csv_file in csv_file_list:
        csv_file_path = csv_folder_path + '/' + csv_file
        with open(csv_file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            first_row = next(reader)
            for row in reader:
                i = 0
                for param in row[1:]:
                    transformation_list[i].append(float(param))
                    i+=1
        plt.boxplot(transformation_list)
        transformation_list = [[],[],[],[],[]]
        plt.title(first_row[0])
        plt.xlabel('Params')
        plt.ylabel('Complexity Score')
        plt.savefig(save_path+first_row[0]+'.png')
        plt.close()

if __name__ == '__main__':
    save_path = './graphs/scatter_plot/'
    box_plot(save_path)

