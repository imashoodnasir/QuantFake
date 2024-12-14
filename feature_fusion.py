
import numpy as np

def feature_fusion(image_features, text_features):
    return np.concatenate((image_features, text_features), axis=1)
