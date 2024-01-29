import cv2
import numpy
from color_utils import BM3D_1st_step_color, BM3D_2nd_step_color
from gray_utils import BM3D_1st_step, BM3D_2nd_step
from noise_estimation import noise_estimate

def is_grayscale(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Check if the image has one channel (grayscale) or three channels (RGB)
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        return True, img
    elif len(img.shape) == 3 and img.shape[2] == 3:
        return False, img
    else:
        # Handle other cases (e.g., images with more than 3 channels)
        raise ValueError("Unsupported number of color channels")

def bm3d(image_path, quality='best', output_format='rgb'):
    
    condition, image = is_grayscale(image_path)
    
    if condition == True:
        step1_image = BM3D_1st_step(image, sigma, quality)
        step2_image = BM3D_2nd_step(step1_image, image, sigma, quality)

        return step2_image
    
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sigma = noise_estimate(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        step1_image = BM3D_1st_step_color(image, sigma, quality)
        step2_image = BM3D_2nd_step_color(step1_image, image, sigma, quality)
        if output_format == 'rgb':
            step2_image = cv2.cvtColor(step2_image, cv2.COLOR_YCrCb2RGB)
        elif output_format == 'bgr':
            step2_image = cv2.cvtColor(step2_image, cv2.COLOR_YCrCb2BGR)
        else:
            raise ValueError("Unsupported output format. Choose 'rgb' for RGB image or 'bgr' for BGR image")

        return step2_image