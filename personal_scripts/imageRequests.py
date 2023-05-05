import requests
import os
import threading

#implementare il multithreading per scaricare più immagini contemporaneamente

categories = ['abstract','advertisement','architecture','object','paintings','person','scene','transport']
photo_number = 3
public_key = 'P7m3-dlFVMxb-Lbtf89vFn5zNYof55mcNkI36L3h2VA'

def download_images(category):
    url = f'https://api.unsplash.com/photos/random?client_id={public_key}&count={photo_number}&query={category}'
    response = requests.get(url)
    with open(f'./my_images/{category}_{name}.jpg', 'wb') as f:
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
                
                f.write(image_response.content)
            except:
                print("Invalid image")

if __name__ == "__main__":
    thread_array = []
    if not os.path.exists("./my_images"):
        os.makedirs("./my_images")
    
    for i in range(0,len(categories)-1):
        thread_array.append(threading.Thread(target=download_images, args=(categories[i],)))
        thread_array[i].start()
        
    for thread in thread_array:
        thread.join()