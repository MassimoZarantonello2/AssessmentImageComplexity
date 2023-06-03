path = './ResNet18/test_results/Savoias_ResNet18_fourth_layer.txt'
path_write = './ResNet18/test_results/new_Savoias_ResNet18_fourth_layer.txt'
last_type = 'Advertisement'
d = {}
# Read the file
with open(path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        name,score = line.split(' ')
        x = name.split('_')
        x_name = x[:len(x)-1]
        number = x[len(x)-1]
        number = int(number[:-4])
        extention = x[len(x)-1][-4:]
        for i in range(len(x_name)):
            if i == 0:
                name = x_name[i]
            else:
                name = name + '_' + x_name[i]
        if name != last_type:
            with open(path_write, 'a') as f:
                for key in sorted(d):
                    f.write(d[key])  
            last_type = name
            d = {}
        d[number] = f'{name}_{number}{extention} {score}'
        i = 0
        
with open(path_write, 'a') as f:
    for key in sorted(d):
        f.write(d[key])