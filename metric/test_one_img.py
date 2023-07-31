import os
import cv2
import numpy as np

def calculate_rgb(image_path):
    
    r_variance = 0
    g_variance = 0
    b_variance = 0
 
    image = cv2.imread(image_path)
          
    image = image.astype(np.float64) / 255.0
    b_channel, g_channel, r_channel = cv2.split(image)

    #r_mean += np.mean(r_channel)
    #g_mean += np.mean(g_channel)
    #b_mean += np.mean(b_channel)

    b_variance += np.var(b_channel)
    g_variance += np.var(g_channel)
    r_variance += np.var(r_channel)
                
    return r_variance, g_variance, b_variance

def calculate_color_mean_and_variance(image_path):
    
    total_pixels = 0
    color_sum = np.array([0, 0, 0], dtype=np.float64)
    color_sum_squared = np.array([0, 0, 0], dtype=np.float64)

    image = cv2.imread(image_path)

    height, width, channels = image.shape
    total_pixels += height * width
        
    image_float = image.astype(np.float64) / 255.0
    color_sum += np.sum(image_float, axis=(0, 1))
    color_mean = color_sum / total_pixels
    color_sum_squared += np.sum(np.square(image_float), axis=(0, 1))
    color_variance = (color_sum_squared / total_pixels) - np.square(color_mean)
    
    return color_mean, color_variance


if __name__ == "__main__":

    
    folder_img = '/cluster/work/cvl/denfan/diandian/control/metric/rabbit_1.jpg'
    r_variance, g_variance, b_variance = calculate_rgb(folder_img)
    print("Color Variance (R, G, B):", r_variance, g_variance, b_variance)
    mean, variance = calculate_color_mean_and_variance(folder_img)
    print("Color Mean (R, G, B):", mean)
    print("Color Variance (R, G, B):", variance)
    
    
    
    folder_img = '/cluster/work/cvl/denfan/diandian/control/metric/rabbit_2.jpg'
    r_variance, g_variance, b_variance = calculate_rgb(folder_img)
    print("Color Variance (R, G, B):", r_variance, g_variance, b_variance)
    mean, variance = calculate_color_mean_and_variance(folder_img)
    print("Color Mean (R, G, B):", mean)
    print("Color Variance (R, G, B):", variance)