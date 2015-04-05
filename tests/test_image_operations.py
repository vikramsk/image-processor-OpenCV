__author__ = 'Vikram'

from unittest import TestCase
import cv2
import os
import shutil
import scripts.image_operations as image_operations
import scripts.image_handler as image_handler

class ImageHandlerTests(TestCase):
    def test_load_image_descriptors_from_folder(self):
        folder_path = "D:\CodeBase\Python CodeBase\Data\\3_Star_Hotels_Images\\313404\EntireSet\\"
        image_descriptors = image_operations.ImageOperations().load_image_descriptors_from_folder(folder_path)
        self.assertGreater(list(image_descriptors).__sizeof__(), 0)

    def test_get_similar_images(self):
        folder_path = "D:\CodeBase\Python CodeBase\Data\\3_Star_Hotels_Images\\313404\EntireSet\\"
        image_operator = image_operations.ImageOperations()
        image_operator.load_image_descriptors_from_folder(folder_path)
        image_path = "D:\CodeBase\Python CodeBase\Data\\3_Star_Hotels_Images\\313404\EntireSet\\8850442.jpg"

        image_file = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        image_descriptor = image_handler.ImageHandler(image_file).get_sift_descriptor()

        similar_images = image_operator.get_similar_image(image_descriptor)
        self.assertGreater(list(similar_images).__sizeof__(), 0)

    def test_get_unique_images(self):
        folder_path = "D:\CodeBase\Python CodeBase\Data\\3_Star_Hotels_Images\\313404\EntireSet\\"
        dest_folder = "G:\\Trash\\"
        image_operator = image_operations.ImageOperations()

        file_list = {}
        files = os.listdir(folder_path)
        for file in files:
            abs_file_path = os.path.abspath(folder_path + file)
            file_list[file] = abs_file_path

        unique_image_set = image_operator.get_unique_image_set(folder_path)

        for file_name  in unique_image_set:
            shutil.copy(file_list[file_name], dest_folder)

        self.assertGreater(list(unique_image_set).__sizeof__(), 0)