import pymongo
import pandas as pd
from secure_db_connection import SecureConnect


class MongoDB:
    def __init__(self, database_name, connection_string):
        self._database_name = database_name
        self._connection_string = connection_string
        secure_connection = SecureConnect()
        self._password = secure_connection.decrypt() # Private variable for storing the password
        self._user_name = secure_connection.get_user_name() # Private variable for storing the username 
        # Private variable for storing the connection string
        self._connection_string = self._connection_string.replace("<username>",self._user_name).replace("<password>",self._password)
        self.client = None

    def connect_to_mongo_db(self):
        """
        Function to connect to MongoDB using connection string. 
        After initiating connection with MongoDB database, it gives feedback about successfull connection
        """
        try:
            if self.client is None:
                self.client = pymongo.MongoClient(self._connection_string)
                print("MongoDB connected")
        except (Exception,) as e:
            print("Error in connecting MongoDB Cloud")

    def get_collection(self):
        """
        Function to return the collection name
        """
        if self.client is not None:
            database = self.client[self._database_name]
            collection = database.ckd_collection  # Collection Name in the Database
            return collection
        else:
            print("Cannot connect with database due to failure in Internet connection")
            return None

    def read_data_from_mongodb(self):
        """
        Function to read data from the MongoDB and convert it into Pandas Dataframe object
        """
        collection = self.get_collection()
        if collection is not None:
            data = pd.DataFrame.from_records(collection.find())
            return data
        else:
            print("Failed to read data from MongoDB Database. Check the internet connectivity")
            return None
