def parseFile(read_path, write_path):
    label_list = []
    label_score_dict = {}
    
    with open(read_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            name,score = line.split("  ", 1)
            label_list.append(name)
            label_score_dict[name] = score
        
        label_list.sort()
        with open(write_path,'w+') as g:
            for name in label_list:
                g.write(name.replace(' ','_') + " " + label_score_dict[name])

if __name__ == '__main__':
    parseFile('./IC9600/test.txt','./IC9600/test_parsed.txt')
    parseFile('./IC9600/train.txt','./IC9600/train_parsed.txt')
    #parseFile('./IC9600/test_and_train.txt','./IC9600/test_and_train_parsed.txt')