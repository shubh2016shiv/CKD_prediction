from mongodb_util import MongoDB
from configparser import ConfigParser
import pickle

"""
NOTE: This is just a utility python file containing miscellaneous functions which are used in project many times
"""

class HelperFunctions:
    def __init__(self):
        self.blog_text_files_dir = 'resources/blog_text_content/'
        self.config = ConfigParser()
        self.config.read('config.ini') # for reading the configurations from the 'config.ini'
        # instantiate the MongoDB class
        self.mongodb = MongoDB(database_name=self.config['mongo']['database_name'],
                               connection_string=self.config['mongo']['connection_string'])

    def write_text_to_blog(self, file_name):
        """
        Function for writing the text on blog from the txt file
        :param file_name: 
        :return: text as String
        """
        with open(self.blog_text_files_dir + file_name, 'r', encoding='utf-8') as reader:
            text = reader.read()
            return text

    def initiate_mongodb_connection(self):
        """
        Function to establish mongoDB connection
        :param: None 
        :return: none 
        """
        self.mongodb.connect_to_mongo_db()

    def get_data_from_mongodb(self):
        """
        Function to get data from MongoDB
        :param: None 
        :return: returns raw data after reading it from MongoDB 
        """
        data = self.mongodb.read_data_from_mongodb()
        if data is not None:
            return data[self.config['dataframe']['column_names'].split(",")]
        else:
            print("Failed to read data from MongoDB Database")

    def save_dataframe_to_csv(self, dataframe, file_name):
        """
        Utility Function saving the dataframe as csv
        :param: dataframe,  file_name
        :return: None 
        """
        dataframe.to_csv(self.config['processed_data']['path'] + file_name, index=False)

    @staticmethod
    def save_as_pickle_file(obj, pickle_name):
        with open(pickle_name + '.pkl', 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def load_pickle_file(pickle_name):
        with open(pickle_name + '.pkl', 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def color_type(val):
        color = 'green' if val == 'Numerical' else 'red'
        return f'background-color: {color}'
