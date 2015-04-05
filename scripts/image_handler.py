__author__ = 'Vikram'
import numpy as np
import cv2
import image_descriptors as i_desc

class ImageHandler:
    def __init__(self, image):
        self.image = image

    def get_descriptor(self, image_descriptor):
        if image_descriptor == i_desc.ImageDescriptors.SIFT:
            return self.get_sift_descriptor()

    def get_sift_descriptor(self):
        detector = cv2.SIFT()
        self.image
        kp, descriptor = detector.detectAndCompute(self.image, None)
        #desc = np.reshape(descriptor, (len(descriptor)/128, 128))
        #desc = np.float32(desc)
        #descriptor = cv2.normalize(descriptor)
        return descriptor