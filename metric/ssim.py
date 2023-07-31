import cv2
from skimage.metrics import structural_similarity as ssim


def calculate_ssim(image1, image2):

    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    ssim_score, _ = ssim(image1_rgb, image2_rgb, multichannel=True, full=True, channel_axis=2)

    return ssim_score

if __name__ == "__main__":

    image1_path = "/cluster/work/cvl/denfan/diandian/control/T2I-COD/test_images/similarity/rabbit_inpainting_1.jpg"
    #image1_path = "/cluster/work/cvl/denfan/diandian/control/T2I-COD/test_images/similarity/COD10K-CAM-2-Terrestrial-40-Rabbit-2359.jpg"

    image2_path = "/cluster/work/cvl/denfan/diandian/control/T2I-COD/test_images/similarity/background.jpg"

    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)


    ssim_score = calculate_ssim(image1, image2)
    print(f"SSIM: {ssim_score}")
