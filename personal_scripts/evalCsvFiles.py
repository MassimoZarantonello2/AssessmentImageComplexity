import csv
import os
import evalResults as evalRes

main_path = './transformation/files_csv/'
test_ICNet_results_path = './ICNet/test_results/test_ICNet_results.txt'
csv_files = os.listdir(main_path)
test_dict = {}
scores = [[],[],[],[],[]]
GT_score = []

with open(test_ICNet_results_path, 'r') as f:
    for line in f:
        name,score = line.split(' ')
        GT_score.append(float(score))
        
for csv_file in csv_files:
    with open(main_path + csv_file,'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            csv_name = row[0]
            csv_scores = row[1:]
            i = 0
            for score in csv_scores:
                scores[i].append(float(score))
                i += 1
                
for i in range(len(scores)):
    print("\nScores for " + str(i+1) + "th class: ")
    print(evalRes.evaInfo(GT_score,scores[i]))
    print('-'*50)