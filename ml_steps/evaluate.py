from sklearn.metrics import classification_report
import pandas as pd
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import PrecisionRecallCurve


class Evaluate:
    def __init__(self,trained_pipeline,train_data, test_data, train_label, test_label):
        self.trained_pipeline = trained_pipeline
        self.X_train = train_data
        self.X_test = test_data
        self.y_train = train_label
        self.y_test = test_label
        self.y_pred = None

    def get_classification_report(self):
        visualizer = ClassificationReport(self.trained_pipeline, classes=['notckd', 'ckd'], support=True)
        visualizer.fit(self.X_train, self.y_train)  # Fit the visualizer and the model
        visualizer.score(self.X_test, self.y_test)  # Evaluate the model on the test data
        return visualizer

    def get_ROCAUC_curve(self):
        visualizer = ROCAUC(self.trained_pipeline, classes=['notckd', 'ckd'])
        visualizer.fit(self.X_train, self.y_train)  # Fit the training data to the visualizer
        visualizer.score(self.X_test, self.y_test)  # Evaluate the model on the test data
        return visualizer

    def get_Precision_Recall_curve(self):
        visualizer = PrecisionRecallCurve(self.trained_pipeline)
        visualizer.fit(self.X_train, self.y_train)  # Fit the training data to the visualizer
        visualizer.score(self.X_test, self.y_test)  # Evaluate the model on the test data
        return visualizer

    def get_confusion_matrix(self):
        visualizer = ConfusionMatrix(self.trained_pipeline, classes=['notckd', 'ckd'])

        visualizer.fit(self.X_train, self.y_train)  # Fit the training data to the visualizer
        visualizer.score(self.X_test, self.y_test)  # Evaluate the model on the test data
        return visualizer

