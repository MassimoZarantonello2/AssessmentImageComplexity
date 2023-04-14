import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type = str, default = './IC9600/images')
parser.add_argument('-i', '--input', type = str, default = './IC9600/images')
parser.add_argument('-o', '--output', type = str, default = './out')



if __name__ == "__main__":
    args  = parser.parse_args()