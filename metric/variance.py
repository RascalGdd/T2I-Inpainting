import os
import cv2
import numpy as np

def get_box(mask_path):
  binary_image = cv2.imread(mask_path)
  nonzero_pixels = np.nonzero(binary_image)
  min_x = np.min(nonzero_pixels[1])
  max_x = np.max(nonzero_pixels[1])
  min_y = np.min(nonzero_pixels[0])
  max_y = np.max(nonzero_pixels[0])  

  return min_x,max_x,min_y,max_y




def calculate_mean_and_variance(folder_img):
    
    total_pixels = 0
    color_sum = np.array([0, 0, 0], dtype=np.float64)
    color_sum_squared = np.array([0, 0, 0], dtype=np.float64)
    
    for filename in os.listdir(folder_img):
        if filename.lower().endswith(('.jpg', '.png')):
            image_path = os.path.join(folder_img, filename)    
            image = cv2.imread(image_path)
    
            height, width, channels = image.shape
            total_pixels += height * width
                
            image_float = image.astype(np.float64) / 255.0
            color_sum += np.sum(image_float, axis=(0, 1))
            color_sum_squared += np.sum(np.square(image_float), axis=(0, 1))
    
    color_mean = color_sum / total_pixels
    color_variance = (color_sum_squared / total_pixels) - np.square(color_mean)
    
    return color_mean, color_variance

def calculate_rgb(folder_img,folder_mask):
    
    r_variance = 0
    g_variance = 0
    b_variance = 0
    num = 0
    for filename in os.listdir(folder_img):
        if filename.lower().endswith(('.jpg', '.png')):
            #img
            image_path = os.path.join(folder_img, filename)    
            image = cv2.imread(image_path)  
            #mask
            mask_path =os.path.join(folder_mask,filename.replace('jpg','png'))
            min_x,max_x,min_y,max_y = get_box(mask_path)
            image = image[min_y:max_y,min_x:max_x,:]      
            image = image.astype(np.float64)/255
            b_channel, g_channel, r_channel = cv2.split(image)
            b_variance += np.var(b_channel)
            g_variance += np.var(g_channel)
            r_variance += np.var(r_channel)
            num += 1

    return r_variance/num, g_variance/num, b_variance/num


if __name__ == "__main__":
    #folder_img = "/cluster/scratch/denfan/syhthetic_cod/"
    #mean, variance = calculate_color_mean_and_variance(folder_img)
    #print("Color Mean (R, G, B):", mean)
    #print("Color Variance (R, G, B):", variance)
    folder_mask = '/cluster/work/cvl/denfan/diandian/control/inpainting/datasets/camo_diff_512/camo_mask'
    
    folder_img = '/cluster/work/cvl/denfan/diandian/control/T2I-COD/experiments/train_color_one/visualization/'
    r_variance, g_variance, b_variance = calculate_rgb(folder_img,folder_mask)
    print("Color Variance (R, G, B):", r_variance, g_variance, b_variance)
    
    folder_img = '/cluster/work/cvl/denfan/diandian/control/T2I-COD/experiments/background_one/'
    r_variance, g_variance, b_variance = calculate_rgb(folder_img,folder_mask)
    print("Color Variance (R, G, B):", r_variance, g_variance, b_variance)
        

    


