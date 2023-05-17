import os
from PIL import Image

test_path = './IC9600/parsed_files/test_parsed.txt'
IC9600_image_path = './IC9600/images/'


images_list = []
test_list = []
with open(test_path, 'r') as GT_test:
    for row in GT_test:
        name,score = row.split(' ',1)
        test_list.append(name)
        
print(len(test_list))
i = 1
images = os.listdir(IC9600_image_path)
for image in images:
    image_name = image.replace(' ','_')
    if image_name in test_list:
        print(f'swapping image: {i}/{len(test_list)}',end='\r')
        im = Image.open(IC9600_image_path+image)
        im.save('./IC9600/test_images/'+image_name)
        i+=1
print(f'swapped: {i}/{len(test_list)}')