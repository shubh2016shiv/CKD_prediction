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
        """
        Get classification report with heatmap
        :return: yellowbrick.classifier.classification_report.ClassificationReport
        """
        visualizer = ClassificationReport(self.trained_pipeline, classes=['notckd', 'ckd'], support=True)
        visualizer.fit(self.X_train, self.y_train)  # Fit the visualizer and the model
        visualizer.score(self.X_test, self.y_test)  # Evaluate the model on the test data
        return visualizer
    
    def get_accuracy(self):
        """
        calculates the accuracy of the model
        :return: accuracy in float
        """
        y_pred = self.trained_pipeline.predict(self.X_test)
        return accuracy_score(self.y_test,y_pred)

    def get_ROCAUC_curve(self):
        """
        Get ROC AUC curve
        :return: yellowbrick.classifier.rocauc.ROCAUC
        """
        visualizer = ROCAUC(self.trained_pipeline, classes=['notckd', 'ckd'])
        visualizer.fit(self.X_train, self.y_train)  # Fit the training data to the visualizer
        visualizer.score(self.X_test, self.y_test)  # Evaluate the model on the test data
        return visualizer

    def get_Precision_Recall_curve(self):
        """
        Get Precision Recall curve
        :return: yellowbrick.classifier.prcurve.PrecisionRecallCurve
        """
        
        visualizer = PrecisionRecallCurve(self.trained_pipeline)
        visualizer.fit(self.X_train, self.y_train)  # Fit the training data to the visualizer
        visualizer.score(self.X_test, self.y_test)  # Evaluate the model on the test data
        return visualizer

    def get_confusion_matrix(self):
        """
        Get confusion matrix
        :return: yellowbrick.classifier.confusion_matrix.ConfusionMatrix'
        """
        
        visualizer = ConfusionMatrix(self.trained_pipeline, classes=['notckd', 'ckd'])

        visualizer.fit(self.X_train, self.y_train)  # Fit the training data to the visualizer
        visualizer.score(self.X_test, self.y_test)  # Evaluate the model on the test data
        return visualizer, visualizer.score_

