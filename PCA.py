import numpy as np
import os
import struct
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from matplotlib.pyplot import imshow
from time import time


def load_mnist(path="/"):

    train_labels_path = os.path.join(path, "train-labels-idx1-ubyte")
    train_images_path = os.path.join(path, "train-images-idx3-ubyte")

    test_labels_path = os.path.join(path, "t10k-labels-idx1-ubyte")
    test_images_path = os.path.join(path, "t10k-images-idx3-ubyte")

    labels_path = [train_labels_path, test_labels_path]
    images_path = [train_images_path, test_images_path]

    labels = []
    images = []

    for path in zip(labels_path, images_path):

        with open(path[0], "rb") as lbpath:
            magic, n = struct.unpack(">II", lbpath.read(8))
            lb = np.fromfile(lbpath, dtype=np.uint8)
            labels.append(lb)

        with open(path[1], "rb") as imgpath:
            magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
            images.append(np.fromfile(imgpath, dtype=np.uint8).reshape(len(lb), 784))

    return images[0], images[1], labels[0], labels[1]


X_train, X_test, Y_train, Y_test = load_mnist(path="MNIST")

imshow(X_test[0].reshape([28, 28]), cmap="gray")

mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

lr = LogisticRegression(max_iter=1000)
start = time()
lr.fit(X_train, Y_train)
end = time()

acc = accuracy_score(Y_test, lr.predict(X_test))
acc_train = accuracy_score(Y_train, lr.predict(X_train))
lloss = log_loss(Y_test, lr.predict_proba(X_test))
lloss_train = log_loss(Y_train, lr.predict_proba(X_train))

print(f"Time: {end - start}s, Acc: {acc} / {acc_train}, Loss: {lloss} / {lloss_train}")

# PCA with min var
pca = PCA(0.90)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

lr = LogisticRegression(max_iter=1000)
start = time()
lr.fit(X_train_pca, Y_train)
end = time()

acc = accuracy_score(Y_test, lr.predict(X_test_pca))
acc_train = accuracy_score(Y_train, lr.predict(X_train_pca))
lloss = log_loss(Y_test, lr.predict_proba(X_test_pca))
lloss_train = log_loss(Y_train, lr.predict_proba(X_train_pca))

print(
    f"PCA - Time: {end - start}s, Acc: {acc} / {acc_train}, Loss: {lloss} / {lloss_train}"
)
