import re

def parse(path):    
    lines = f.readlines()
    for line in lines:
        category,x = line.split("_", 1)
        name,cpl_score = x.split("  ", 1)
        f2.write(category + " " + name + " " + cpl_score)
            


            
if __name__ == "__main__":
    path = "./IC9600/train.txt"
    path1 = "./IC9600/test.txt"
    with open("./IC9600/GT_splitted.txt", 'a') as f2:
        with open(path, 'r') as f:
            with(path1,'r') as f1:
                parse(path)
                parse(path1)