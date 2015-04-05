__author__ = 'Vikram'

from unittest import TestCase
import cv2
import scripts.image_handler as image_handler

class ImageHandlerTests(TestCase):
    def test_get_sift_descriptor(self):
        image_path = ""
        image_file = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        i_handler = image_handler.ImageHandler(image_file)
        sift_descriptors = i_handler.get_sift_descriptor()
        self.assertIsNotNone(sift_descriptors)