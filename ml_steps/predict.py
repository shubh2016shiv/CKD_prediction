import sklearn
import pandas as pd


class Predict:
    def __init__(self, pipeline:sklearn.pipeline.Pipeline, new_data:pd.core.frame.DataFrame):
        self.pipeline = pipeline
        if len(new_data) == 1:
            self.new_data = new_data
        self._classes = ['Chronic Kidney Disease NOT Detected','Chronic Kidney Disease Detected']

    def show_prediction(self):
        """
        Perform the prediction from the given pipeline
        :return: prediction results
        """
        y_pred = self.pipeline.predict(self.new_data)

        prediction = self._classes[y_pred[0]]
        return prediction
