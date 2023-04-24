import requests
import json
import os
import threading

#implementare il multithreading per scaricare più immagini contemporaneamente
#implementare il multithreading per scaricare categorie contemporaneamente

categories = ['scene','object','paintings']
photo_number = 2
public_key = 'P7m3-dlFVMxb-Lbtf89vFn5zNYof55mcNkI36L3h2VA'

def download_images(category):
    url = f'https://api.unsplash.com/photos/random?client_id={public_key}&count={photo_number}&query={category}'
    response = requests.get(url)
    for image in response.json():
        try:
            image_url = image['urls']['small_s3']
            image_response = requests.get(image_url)
            if image['alt_description'] is not None:
                name = image['alt_description'].replace(' ', '_')
            elif image['tags'][0]['souce']['cover_photo']['alt_description'] is not None:
                name = image['tags'][0]['souce']['cover_photo']['alt_description'].replace(' ', '_')
            else:
                name = image['id']  
            with open(f'./my_images/{category}_{name}.jpg', 'wb') as f:
                f.write(image_response.content)
        except:
            print("Invalid image")

if __name__ == "__main__":

    if not os.path.exists("./my_images"):
        os.makedirs("./my_images")
        thread_array = []
    
    for i in range(0,len(categories)):
        thread_array[i] = threading.Thread(target=download_images, args=(categories[i],))
        thread_array[i].start()
        
    for thread in thread_array:
        thread.join()