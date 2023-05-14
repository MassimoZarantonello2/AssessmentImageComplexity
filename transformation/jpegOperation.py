import torchvision.transforms as T
import numpy as np
from PIL import Image

#create a global variable for Q matrix
Q_matrix = [[16,11,10,16,24,40,51,61],
            [12,12,14,19,26,58,60,55],
            [14,13,16,24,40,57,69,56],
            [14,17,22,29,51,87,80,62],
            [18,22,37,56,68,109,103,77],
            [24,35,55,64,81,104,113,92],
            [49,64,78,87,103,121,120,101],
            [72,92,95,98,112,100,103,99]]

def compute_DCT(channel,compression_rate):
    global Q_matrix
    results = np.zeros((channel.shape[0],channel.shape[1]))
    if(compression_rate >0.5):
        Q_matrix = np.multiply(Q_matrix,(1-compression_rate)/0.5)
    else:
        Q_matrix = np.multiply(Q_matrix,0.5/(compression_rate))
        
    Q_matrix = np.round(Q_matrix)
    Q_matrix = np.clip(Q_matrix,1,255)
    for x in range(0,channel.shape[0],8):
        for y in range(0,channel.shape[1],8):
            M_matrix = channel[x:x+8,y:y+8]
            T_matrix = compute_textel(M_matrix)
            T_matrix_transpose = np.transpose(T_matrix)
            D_matrix = np.matmul(T_matrix,np.matmul(M_matrix,T_matrix_transpose))
            C_matrix = np.round(np.divide(D_matrix,Q_matrix))
            C_matrix = np.int16(C_matrix)
            results[x:x+8,y:y+8] = C_matrix
    return results               
            
def compute_textel(textel):
    textel = np.float16(textel)
    for x in range(0,len(textel)):
        for y in range(0,len(textel)):
            if x == 0:
                textel[x,y] = np.round(1/np.sqrt(8),4)
            else:
                textel[x,y] = np.round((np.sqrt(0.25) * np.cos((2*y+1)*x*np.pi/16)),4)      
    return textel
    
def compute_jpeg_compression_algorithm(channel,compression_rate=0.5):
    channel = np.int16(channel)-128
    compressed_matrix = compute_DCT(channel,compression_rate)
    return compressed_matrix

#-----------------------------------------------------------------DECOMPRESS JPEG---------------------------------------------------------------------------------
def compute_inverse_jpeg_compression(channel):
    R_matrix = multiplty_textels(channel)
    decompressed_matrix = compute_IDCT(R_matrix)
    R_matrix = np.int16(decompressed_matrix)
    R_matrix = np.clip(R_matrix, 1, 255)
    return R_matrix

def multiplty_textels(channel):
    x_dim,y_dim = len(channel),len(channel[0])
    channel = np.array(channel)
    global Q_matrix
    results = np.zeros((x_dim,y_dim))
    for x in range(0,x_dim,8):
        for y in range(0,y_dim,8):
            results[x:x+8,y:y+8] = np.multiply(channel[x:x+8,y:y+8],Q_matrix)
    return np.round(results)

def compute_IDCT(channel):
    results = np.zeros((channel.shape[0],channel.shape[1]))
    for x in range(0,channel.shape[0],8):
        for y in range(0,channel.shape[1],8):
            R_matrix = channel[x:x+8,y:y+8]
            T_matrix = compute_textel(R_matrix)
            T_matrix_transpose = np.transpose(T_matrix)
            N_matrix = np.round(np.matmul(T_matrix_transpose,np.matmul(R_matrix,T_matrix)))+128
            results[x:x+8,y:y+8] = N_matrix
    return results


if __name__ == '__main__':

    input_mage = Image.open('./my_images/scene_Twin_Tower,_Malaysia.jpg')
    im_ycbcr = input_mage.convert('YCbCr')
    im_data = np.array(input_mage)

    h, w, c = im_data.shape
    h_new = int(np.ceil(h / 8) * 8)
    w_new = int(np.ceil(w / 8) * 8)
    im_new = np.zeros((h_new, w_new, c), dtype=np.uint8)

    im_new[:h, :w, :] = im_data
    im_resized = Image.fromarray(im_new).resize((w_new, h_new))
    
    image = np.array(im_resized)
    jpeg_factor = 0.9
    jpeg_image = np.zeros((h_new,w_new,3),dtype=np.uint8)
    for i in range(0,3):
        channel = image[:,:,i]
        output = compute_jpeg_compression_algorithm(channel,jpeg_factor)
        output = compute_inverse_jpeg_compression(output)
        jpeg_image[:,:,i] = output
    
    image = Image.fromarray(jpeg_image)
    image_RGB = image.convert('RGB')
    image.show()
