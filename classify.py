from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import numpy as np


X, y = fetch_openml('mnist_784', version = 1, return_X_y = True)


xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=2500, train_size=7500, random_state = 9)
xtrainscaled = xtrain/255.0
xtestscaled = xtest/255.0

clf = LogisticRegression(solver = 'saga',multi_class='multinomial').fit(xtrainscaled, ytrain)

        
def get_prediction(image):

    img_pil = Image.open(image)

    #Converting to grayscale - L means each pixel is denoted by one number
    img_bw = img_pil.convert('L')
    img_resize = img_bw.resize((28,28),Image.ANTIALIAS)
    pix_filter = 20
    min_pix = np.percentile(img_resize, pix_filter)
    img_scale = np.clip(img_resize-min_pix,0,255)
    max_pix = np.max(img_resize)
    img_scale = np.asarray(img_scale)/max_pix

    test_sam = np.array(img_scale).reshape(1, 784)
    test_predict = clf.predict(test_sam)
    return test_predict[0]

