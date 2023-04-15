import re
        
def generate_list_dict (path,lista,dict):
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            name,cpl_score = line.split("  ", 1)
            name = name.replace(" ","_")
            lista.append(name)
            print(name)
            dict[name] = (cpl_score)
    return lista,dict

                     
            
if __name__ == "__main__":
    path = "./IC9600/train.txt"
    path1= "./IC9600/test.txt"
    GT_dict = {}
    GT_names = []
    
    GT_names,GT_dict = generate_list_dict(path,GT_names,GT_dict)
    GT_names,GT_dict = generate_list_dict(path1,GT_names,GT_dict)
    
    GT_names.sort()
    with open("./IC9600/GT_IC9600.txt", 'w') as f2:
        for i in range(len(GT_names)):
            name = GT_names[i]
            f2.write(name + " " + GT_dict[name])