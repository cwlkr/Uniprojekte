import numpy as np
import os.path
from PIL import Image
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from collections import namedtuple
from utils import basename_without_ext
from transcription import transcription
from features import Features


# Wrapper structure to keep track of sequence of feature vectors and the
# path/id of the image they were extracted from.
ImageData = namedtuple('Image', ('path', 'id', 'label', 'feature_vectors'))

class KeywordSpotter:
    """
    Returns images with similar words in them given a query image with a word.
    """

    def __init__(self, image_paths):
        """
        :param image_paths: list of file paths 
        """
        self.images = list(map(self.__build_image, image_paths))

    def spot(self, query_image_path):
        """
        Finds images with words that are similar to the word in query_image.
        
        :param query_image (str): path to query image
        :return: list word images 
        """
        self.query_image = self.__build_image(query_image_path)
        vectors = self.query_image.feature_vectors


        return list(filter(
            lambda i: self.__dtw(vectors, i.feature_vectors) <= 25,
            self.images
        ))

    def __build_image(self, image_path):
        """
        Extracts sequence of feature vectors out of image and turns it
        into an Image.
        
        :param image (str): path to a word image 
        :return: Image
        """
        vectors = self.__extract_feature_vectors(image_path)
        id = basename_without_ext(image_path)
        label = transcription(id)
        return ImageData(
            path=image_path,
            id=id,
            label=label,
            feature_vectors=vectors
        )

    def __extract_feature_vectors(self, image):
        dirname = os.path.dirname(os.path.abspath(__file__))
        im = Image.open(os.path.join(dirname, image))

        pixels = list(im.getdata())
        width, height = im.size
        pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]

        return Features.get_features(pixels)

    def __dtw(self, feature_vectors_1, feature_vectors_2):
        """
        :param feature_vectors_1: Sequence of feature vectors of first image 
        :param feature_vectors_2: Sequence of feature vectors of second image
        :return (float): DTW distance
        """

        distance = fastdtw(
            np.array(feature_vectors_1),
            np.array(feature_vectors_2),
            dist=euclidean
        )[0]

        # print(distance)

        return distance



# For testing purposes. Run this file as a script and pass a path to a word
# image to get a list of similar word images.
if __name__ == "__main__":
    import sys
    import glob

    image_path = sys.argv[-1]

    path = "ground-truth/word-images_preprocessed/*.png"
    image_paths = glob.glob(path)
    s = KeywordSpotter(image_paths[:1000])
    images = s.spot(image_path)

    list(filter(lambda i: print(i.path + " (" + i.label + ")"), images))

