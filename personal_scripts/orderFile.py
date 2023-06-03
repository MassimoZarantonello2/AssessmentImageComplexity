path = './VGG-16/test_results/IC9600_VGG-16_fourth_layer_normalized.txt'
test_path = './IC9600/parsed_files/test_parsed.txt'

def create_dict(path):
    d = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            name,score = line.split(' ')
            d[name] = float(score)
        
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

a = create_dict('./VGG-16/test_results/Savoias_VGG16_fourth_layer.txt')

a = sorted(a.items(), key=lambda x: x[0])
write__dict('./VGG-16/test_results/new_Savoias_VGG16_fourth_layer.txt', a)