import cv2
import numpy as np
import scipy
from scipy.misc import imread
import imageio
import cloudpickle as pickle
import random
import os
import matplotlib.pyplot as plt
import time


# Feature extractor
def extract_features(image_path, vector_size=32):
   
    feature_time = time.time()
    image = imageio.imread(image_path, pilmode="RGB")
    try:
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        alg = cv2.ORB_create()
        # Dinding image keypoints
        kps = alg.detect(image)
        # Getting first 32 of them. 
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print('Error: ', e)
        return None

    print("---Time elapsed in Extracting:  %s seconds ---" % (time.time() - feature_time)) 
    return dsc


def batch_extractor(images_path, pickled_db_path="features.pck"):
    
   # image_path='C:\\Users\\uddipaan.hazarika\\Desktop\\MISC\\Img_Rscr\\'
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]

    result = {}
    for f in files:
        print('Extracting features from image %s' % f)
        name = f.split('/')[-1].lower()
        result[name] = extract_features(f)
        
    #type(a)
    #print(a)
    
    # saving all our feature vectors in pickled file
    with open(pickled_db_path, "wb") as fp:
        pickle.dump(result, fp)
       


      
#Matcher      
     
class Matcher(object):
     
      
    def __init__(self, pickled_db_path="features.pck"):
        with open(pickled_db_path, "rb") as fp:
            self.data = pickle.load(fp)
        self.names = []
        self.matrix = []
        for k, v in self.data.items():
            self.names.append(k)
            self.matrix.append(v)
        self.matrix = np.array(self.matrix)
        self.names = np.array(self.names)

    def cos_cdist(self, vector):
        # getting cosine distance between search image and images database
        v = vector.reshape(1, -1)
        return scipy.spatial.distance.cdist(self.matrix, v, 'cosine').reshape(-1)

    def match(self, image_path, topn=6):
        start_time = time.time()
        features = extract_features(image_path)
        img_distances = self.cos_cdist(features)
        # getting top 5 records
        nearest_ids = np.argsort(img_distances)[:topn].tolist()
        nearest_img_paths = self.names[nearest_ids].tolist()
        
        print("---Time elapsed in matching:  %s seconds ---" % (time.time() - start_time)) 
        return nearest_img_paths, img_distances[nearest_ids].tolist()
    
        
    

#Run
def show_img(path):
    img = imageio.imread(path, pilmode="RGB")
    plt.imshow(img)
    plt.show()
    
def run():
    images_path='C:\\Users\\uddipaan.hazarika\\Desktop\\MISC\\Img_Rscr\\'
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    # getting 3 random images 
    sample = random.sample(files, 6)
    
    batch_extractor(images_path)
    ma = Matcher('features.pck')
    
 
    for s in sample:
        print('Query image ==========================================')
        show_img(s)
        names, match = ma.match(s, topn=6)
        print('Result images ========================================')
        for i in range(6):
            # we got cosine distance, less cosine distance between vectors
            # more they similar, thus we subtruct it from 1 to get match value
            print('Match %s' % (1-match[i]))
            show_img(os.path.join(images_path, names[i]))
    

run()
