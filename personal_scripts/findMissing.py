def create_list(path):
    l = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            name,_ = line.split(' ')
            l.append(name)
        
    return l


path = './VGG-16/test_results/new_Savoias_VGG16_fourth_layer.txt'
path_write = './VGG-16/test_results/new_Savoias_VGG16_fourth_layer_sorted.txt'

d1 = create_list(path)
d2 = create_list(path_write)

for x in d1:
    if x not in d2:
        print(x)