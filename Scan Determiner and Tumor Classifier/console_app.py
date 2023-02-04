import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from skimage.feature import hog
from tensorflow.keras import utils
import tensorflow as tf
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

sift = cv2.SIFT.create()
S1Model = joblib.load(r"C:\Users\beboz\Computer Vision Project\CV2023CSYSDataset\Stage1Gboost.joblib")
S2BrainModel = joblib.load(r"C:\Users\beboz\Computer Vision Project\CV2023CSYSDataset\HoGwithAdaboost.joblib")
S2BreastModel = tf.keras.models.load_model(r"C:\Users\beboz\Computer Vision Project\CV2023CSYSDataset\BreastCNN")
BOW = joblib.load(r"C:\Users\beboz\Computer Vision Project\CV2023CSYSDataset\Stage1BOW.joblib")
codebook = BOW.cluster_centers_
k = len(BOW.cluster_centers_)
def Feature_extractor(image, extractor):
    kp, des = extractor.detectAndCompute(image, None)
    return kp, des
def Hog_Feature_extractor(image):
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(32, 32),cells_per_block=(2, 2), visualize=True, multichannel=True)
    return fd, hog_image
def FreqVecCreator(des):
    visual_words, distance = vq(des, codebook)
    frequency_vector = np.zeros(k)
    for word in visual_words:
        frequency_vector[word] += 1
    return frequency_vector
def BatchFreqVecCreator(features_list):
    visual_words = []
    for image, kp, des in features_list[:]:
        im_visual_words, distance = vq(des, codebook)
        visual_words.append(im_visual_words)
    frequency_vectors = []
    for img_visual_words in visual_words:
        img_frequency_vector = np.zeros(k)
        for word in img_visual_words:
            img_frequency_vector[word] += 1
        frequency_vectors.append(img_frequency_vector)
    return np.stack(frequency_vectors)

def SingleImageClassifier(image_path):
    type = ""
    diagnose = ""
    img = cv2.resize(cv2.cvtColor(cv2.imread(image_path,0), cv2.COLOR_BGR2RGB), (300,300))
    kp, des = Feature_extractor(img, sift)
    x = FreqVecCreator(des)
    x= x.reshape(1,-1)
    x = (x - x.mean()) / x.std()
    y_pred = S1Model.predict(x)
    type = y_pred[0]
    print(y_pred[0])
    if(y_pred == "brain"):
        img = cv2.resize(cv2.imread(image_path), (500,500))
        fd, hog_image = Hog_Feature_extractor(img)
        y_pred = S2BrainModel.predict(fd.reshape(1,-1))
        print(y_pred[0])
        diagnose = y_pred[0]
    elif(y_pred == "breast"):
        # print("wait")
        img = tf.image.resize(cv2.imread(image_path),(256, 256))
        y_pred = S2BreastModel.predict(np.expand_dims(img / 255 , 0)).tolist()
        label_encoded = y_pred[0].index(max(y_pred[0]))
        if label_encoded == 0:
            print("benign")
            diagnose = "benign"
        elif label_encoded == 1:
            print("malignant")
            diagnose = "malignant"
        elif label_encoded == 2:
            print("normal")
            diagnose = "normal"
        else:
            print("error occured")
    return (type, diagnose)


def BatchClassifier():
    #Testing Stage 1 Classifier

    test_df = pd.read_csv(r"../test.csv")
    y_test = test_df['type']
    x_test = []
    for path in test_df["path"]:
        x_test.append(cv2.resize(cv2.cvtColor(cv2.imread("." + path, 0), cv2.COLOR_BGR2RGB), (300, 300)))
    x_test = pd.Series(x_test)
    test_features_list = []
    for image in x_test:
        kp, des = Feature_extractor(image, sift)
        test_features_list.append((image, kp, des))
    new_test_df = pd.DataFrame(data=BatchFreqVecCreator(test_features_list))
    new_test_df = (new_test_df - new_test_df.mean()) / new_test_df.std()
    new_test_df['label'] = y_test
    new_test_df = shuffle(new_test_df)
    X_Test = new_test_df.loc[:, new_test_df.columns != 'label']
    Y_Test = new_test_df['label']
    y_pred = S1Model.predict(X_Test)
    print("Stage 1 Accuracy: " + str(accuracy_score(y_pred, Y_Test)))

    #Testing Stage 2 Brain Classifier

    test_df = pd.read_csv("../brain_test.csv")
    y_test = test_df['diagnose']
    x_test = []
    for path in test_df["path"]:
        x_test.append(cv2.resize(cv2.imread("." + path), (500, 500)))
    x_test = pd.Series(x_test)
    test_hog_features = []
    for image in x_test:
        fd, hog_image = Hog_Feature_extractor(image)
        test_hog_features.append(fd)
    new_test_df = pd.DataFrame(data=test_hog_features)
    new_test_df['label'] = test_df['diagnose']
    new_test_df = shuffle(new_test_df)
    X_Test = new_test_df.loc[:, new_test_df.columns != 'label']
    Y_Test = new_test_df['label']
    y_pred = S2BrainModel.predict(X_Test)
    print("Stage 2 Brain Accuracy: " + str(accuracy_score(y_pred, Y_Test)))

    #Testing Stage 2 Breast Classifier

    test_data = utils.image_dataset_from_directory(r"../Breast scans test")
    test_data = test_data.map(lambda x, y: (x / 255, y))
    test_data = test_data.take(len(test_data))
    accuracy = tf.keras.metrics.Accuracy()
    y_test = []
    x_test = []
    for batch in test_data.as_numpy_iterator():
        x, y = batch
        for ex in y:
            y_test.append(ex)
        for ex in x:
            x_test.append(ex)
        y_pred = S2BreastModel.predict(x)
    loss, accuracy = S2BreastModel.evaluate(np.array(x_test), np.array(y_test))
    print("Stage 2 Breast Accuracy: " + str(accuracy))

def ReadImage():
    for widget in window.winfo_children():
        widget.destroy()
    button = Button(text="Select Scan", command=ReadImage)
    button.pack()
    path = filedialog.askopenfilename()
    type, diagnose = SingleImageClassifier(path)
    display = "Scan Type: " + type + " | Diagnose: " + diagnose
    label = Label(window, text=display).pack()
    img = Image.open(path)
    img = ImageTk.PhotoImage(img)
    img_label = Label(window, image=img,text=display, fg="white")
    # img_label.configure(image = img)
    img_label.image = img
    img_label.pack()


window = Tk()
window.title("Scan Type Determiner and Tumor Classifier")
# window.geometry("600x600")
button = Button(text="Select Scan", command=ReadImage)
button.pack()
window.mainloop()

BatchClassifier()
print("Test Image 1:")
SingleImageClassifier(r"C:\Users\beboz\Computer Vision Project\CV2023CSYSDataset\Brain scans\No tumor\Test\no553.jpg")
print("Test Image 2:")
SingleImageClassifier(r"C:\Users\beboz\Computer Vision Project\CV2023CSYSDataset\Brain scans\Tumor\TEST\y730.jpg")
print("Test Image 3:")
SingleImageClassifier(r"C:\Users\beboz\Computer Vision Project\CV2023CSYSDataset\Breast scans\normal\Test\normal (121).png")
print("Test Image 4:")
SingleImageClassifier(r"C:\Users\beboz\Computer Vision Project\CV2023CSYSDataset\Breast scans\benign\Test\benign (403).png")
print("Test Image 5:")
SingleImageClassifier(r"C:\Users\beboz\Computer Vision Project\CV2023CSYSDataset\Breast scans\malignant\Test\malignant (204).png")
