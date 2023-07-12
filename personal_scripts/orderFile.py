path = './VGG-16/test_results/IC9600_VGG-16_fourth_layer_normalized.txt'

def create_dict(path):
    d = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            name,score = line.split(' ')
            name = name.split('_')[1]
            d[name] = line
        
    return d

def create_list(path):
    l = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            name,_ = line.split(' ')
            l.append(name)
        
    return l

def write__dict(path, d):
    with open(path, 'w') as f:
        for key in d:
            f.write(key[0] + ' ' + str(key[1]) + '\n')

a = create_dict('./my_ICNet_SAVOIAS_results.txt')

a = sorted(a.items(), key=lambda x: x[0])
write__dict('./new_my_ICNet_SAVOIAS_results.txt', a)