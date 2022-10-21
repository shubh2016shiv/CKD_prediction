from configparser import ConfigParser
import numpy as np


class ImportData:
    def __init__(self, mongoDB_connection):
        self.mongoDB_connection = mongoDB_connection
        self.data = None

    def get_data_from_mongoDB(self):
        """
        retrieve data from mongodb into pandas dataframe
        :return: pandas dataframe
        """
        self.data = self.mongoDB_connection.read_data_from_mongodb()  # Read data from MongoDB after successful connection
        
        config = ConfigParser()  # For reading column names from config files, so that column names can be added after fetching data from mongoDB
        config.read('config.ini')  # read column names from the configuration file
        if self.data is not None:
            self.data = self.data[config['dataframe']['column_names'].split(",")]
        else:
            print("Failed to read data from MongoDB Database")

        self.data['classification'] = self.data['classification'].apply(lambda x: x.replace("\t", ""))

        self.data['pcv'] = self.data['pcv'].replace(['\t?'], np.nan).replace(['\t43'], '43')
        self.data['pcv'] = self.data['pcv'].apply(lambda x: float(x))

        self.data['wc'] = self.data['wc'].replace(['\t?'], np.nan).replace(['\t6200'], '6200').replace(['\t8400'],
                                                                                                       '8400')
        self.data['wc'] = self.data['wc'].apply(lambda x: float(x))

        self.data['rc'] = self.data['rc'].replace(['\t?'], np.nan)
        self.data['rc'] = self.data['rc'].apply(lambda x: float(x))

        self.data['dm'] = self.data['dm'].replace(['\tno'], 'no').replace(['\tyes'], 'yes').replace([' yes'], 'yes')

        self.data['cad'] = self.data['cad'].replace(['\tno'], 'no')

        return self.data
