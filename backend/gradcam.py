import cv2
import numpy as np
import os

def generate_gradcam(image_path):

    image = cv2.imread(image_path)
    heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    save_path = os.path.join("reports","heatmap_"+os.path.basename(image_path))

    cv2.imwrite(save_path, overlay)

    return save_path