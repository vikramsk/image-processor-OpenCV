__author__ = 'Vikram'

import os
import cv2
import image_handler
import itertools
import numpy as np

class ImageOperations:
    def load_image_descriptors_from_folder(self, folder_path):
        self.image_indices = {}
        self.image_sizes = {}
        self.image_descriptors = []
        files = os.listdir(folder_path)
        for file in files:
            abs_file_path = os.path.abspath(folder_path + file)
            image_file = cv2.imread(abs_file_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            image_descriptor = image_handler.ImageHandler(image_file).get_sift_descriptor()
            self.image_descriptors.append(image_descriptor)
            self.image_indices[file] = image_descriptor
            self.image_sizes[file] = self.get_image_size(image_file)
        return self.image_descriptors

    def get_image_size(self, image_file):
        height, width = image_file.shape[:2]
        return height + width

    def get_similar_image(self, image_descriptor):
        results = {}
        norm = cv2.NORM_L2
        matcher = cv2.BFMatcher(norm)
        for (file, descriptors) in self.image_indices.items():
            raw_matches = matcher.knnMatch(descriptors, image_descriptor, k=2)
            good = []
            for m,n in raw_matches:
                if m.distance < 0.75* n.distance:
                    good.append([m])
            results[file] = len(good)

        results = sorted([(v, k) for (k, v) in results.items()], cmp = None, key = None, reverse= True)
        return results

    def get_unique_image_set(self, folder_path, distance = 50):
        self.load_image_descriptors_from_folder(folder_path)
        images_to_be_deleted = set()
        for(file, descriptor) in self.image_indices.items():
            if(images_to_be_deleted.__contains__(file)):
                continue
            image_similarity = self.get_similar_image(descriptor)
            best_image_file = file
            for (image_distance, image_file) in image_similarity:
                if (image_file == file or images_to_be_deleted.__contains__(image_file)):
                    continue
                if image_distance > distance:
                    current_best_image = image_file if self.image_sizes[image_file] > self.image_sizes[best_image_file] else best_image_file
                    if image_file == current_best_image:
                        images_to_be_deleted.add(best_image_file)
                        best_image_file = current_best_image
                        continue
                    images_to_be_deleted.add(image_file)
                else:
                    break

        for image_file in images_to_be_deleted :
            del self.image_indices[image_file]

        return self.image_indices.keys()