def create_lista(path,lista):
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            category,x = line.split("_", 1)
            name,cpl_score = x.split("  ", 1)
            lista.append(name)
    return lista

def find_duplicates(lista):
    lista.sort()
    for i in range(len(lista)-1):
        if lista[i] == lista[i+1]:
            print(lista[i])

if __name__ == "__main__":
    path = "./IC9600/train.txt"
    path1= "./IC9600/test.txt"
    
    lista = []
    lista = create_lista(path,lista)
    find_duplicates(lista)