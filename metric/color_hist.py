import cv2
import numpy as np



def calculate_color_histogram(background_folder,inpainting_folder):
    diff = 0
    
    for filename in os.listdir(background_folder):
        if filename.lower().endswith(('.jpg', '.png')):
            #background
            image_path = os.path.join(background_folder, filename)  
            background_image = cv2.imread(image_path)
            lab_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2Lab)
            hist = cv2.calcHist([lab_image], [1, 2], None, [128, 128], [0, 256, 0, 256])
            background_hist = cv2.normalize(hist, hist).flatten()
            #inpainting
            image_path = os.path.join(inpainting_folder, filename)  
            inpainting_image = cv2.imread(image_path)
            lab_image = cv2.cvtColor(inpainting_image, cv2.COLOR_BGR2Lab)
            hist = cv2.calcHist([lab_image], [1, 2], None, [128, 128], [0, 256, 0, 256])
            inpainting_hist = cv2.normalize(hist, hist).flatten()   
            #vs
            diff += cv2.compareHist(background_hist, inpainting_hist, cv2.HISTCMP_CORREL)
            num += 1
            
    diff_hist = diff/num
    return diff_hist


if __name__ == "__main__":

    background_folder='/cluster/work/cvl/denfan/diandian/control/T2I-COD/experiments/background_one/'
    inpainting_folder='/cluster/work/cvl/denfan/diandian/control/T2I-COD/experiments/train_color_one/visualization/'
    diff_hist = calculate_color_histogram(background_folder,inpainting_folder)
