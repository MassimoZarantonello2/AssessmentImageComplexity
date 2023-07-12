
def parse_file(path,write_path):
    d = {}
    last_name = 'Advertisement'
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            name,score = line.split(' ')
            name,index = name.split('_')
            index = index.split('.')[0]
            index = int(index)
            if name == last_name:
                d[index] = line
            else:
                d = sorted(d.items(), key=lambda x: x[0])
                with open(write_path, 'a') as f:
                    for key in d:
                        f.write(key[1])
                d = {}
                last_name = name
                d[index] = line
        d = sorted(d.items(), key=lambda x: x[0])
        with open(write_path, 'a') as f:
            for key in d:
                f.write(key[1])
                
if __name__ == '__main__':
    parse_file('./my_ICNet_SAVOIAS_results.txt','./new_my_ICNet_SAVOIAS_results.txt')