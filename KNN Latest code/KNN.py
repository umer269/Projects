import numpy as np
import sklearn.neighbors
from mnist import MNIST
import sklearn.metrics as skm
import seaborn as sborn
import matplotlib.pyplot as plot
import os
from PIL import Image

class KNN:

    def __init__(self):
        self.mndata = MNIST('Data')
        k = 3
        self.knn = sklearn.neighbors.KNeighborsClassifier(k)

    def load_training_data(self):
        images, labels = self.mndata.load_training()
        return images, labels

    def load_testing_data(self):
        images, labels = self.mndata.load_testing()
        return images, labels

    def train_data(self):
        training_images, training_labels = self.load_training_data()
        self.knn.fit(training_images, training_labels)

    def test_data(self):
        test_images, test_labels = self.load_testing_data()
        predictions = self.knn.predict(test_images)
        test_labels = np.array(test_labels).astype(float)
        predictions = np.array(predictions).astype(float)

        self.save_misclassified_images(test_images, test_labels, predictions)
        self.plot_confusion_matrix(test_labels, predictions)

        print('Accuracy: ' + str(skm.accuracy_score(test_labels, predictions)))
        print('Recall: ' + str(skm.recall_score(test_labels, predictions, average='macro')))
        print('Precision: ' + str(skm.precision_score(test_labels, predictions, average='macro')))
        print('F-Score: ' + str(skm.f1_score(test_labels, predictions, average='macro')))

    def plot_confusion_matrix(self, test_labels, predictions):
        matrix = skm.confusion_matrix(test_labels, predictions)
        plot.subplots(figsize=(11, 6))
        plot.title("Confusion Matrix")
        sborn.heatmap(matrix, annot=True, fmt='g')
        plot.xlabel("Predictions")
        plot.ylabel("Actual values")
        plot.show()

    def save_misclassified_images(self,images,labels,predictions):
        if not os.path.exists('Misclassfied Images'):
            os.mkdir('Misclassfied Images')

        count = 0
        for image, prediction, label in zip(images, predictions, labels):
            if prediction != label:
                image_name = 'Misclassfied Images/' + str(count) + '.png'
                print(image_name + ' classified as: ', prediction, 'but should be: ', label)
                image_temp = np.array(image, dtype='float')
                pixels = image_temp.reshape((28, 28))
                image_resized = Image.fromarray(pixels)
                image_resized = image_resized.convert('RGB')
                image_resized.save(image_name)
                count = count + 1

                #if you want to display these images here oteherwise saved in folder: Misclassified Images
                #plot.imshow(pixels, cmap='gray')
                #plot.show()

