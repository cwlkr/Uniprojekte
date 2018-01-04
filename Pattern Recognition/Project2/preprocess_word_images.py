import os
import numpy as np
import scipy.misc
from skimage import io
from skimage.filters import threshold_otsu


imgDirectory = os.path.join(os.getcwd(), 'ground-truth/word-images')
saveDirectory = os.path.join(os.getcwd(), 'ground-truth/word-images_preprocessed')

for filename in os.listdir(imgDirectory):
    if filename.endswith(".png"):
            currentImg = os.path.join(imgDirectory,filename)
            saveImg = os.path.join(saveDirectory,filename)
            
            #Elimination of greyscale
            wordPicture = io.imread(currentImg, as_grey=1)

            #Automatic clustering and binarization
            thresh_otsu = threshold_otsu(wordPicture)
            binary_otsu = wordPicture > thresh_otsu
            print(binary_otsu)
            print('\nshape: ', binary_otsu.shape)

            #delete white lines on borders
            for _ in range(2):
                #first iteration: delete white lines from top and bottom
                #second iteration: delete white lines from left and right (with help of the transpose)
                
                #delete white lines from top(1.Iteration) & left(2.Iteration)
                numberOfDeletions = 0
                for pxRow in binary_otsu:
                    if False in pxRow:
                        break
                    else:
                        numberOfDeletions += 1
                binary_otsu = binary_otsu[numberOfDeletions:]

                #delete white lines from bottom & right
                numberOfDeletions = 0
                for pxRow in reversed(binary_otsu):
                    if False in pxRow:
                        break
                    else:
                        numberOfDeletions += 1
                binary_otsu = binary_otsu[:len(binary_otsu)-numberOfDeletions]
                
                binary_otsu = np.transpose(binary_otsu)

            print('After deletion: \n', binary_otsu)
            print('\nshape: ', binary_otsu.shape)
            scipy.misc.imsave(saveImg, binary_otsu)
            