import pymongo
import pandas as pd


class MongoDB:
    def __init__(self, database_name, connection_string):
        self.database_name = database_name
        self.connection_string = connection_string
        self.client = None

    def connect_to_mongo_db(self):
        try:
            if self.client is None:
                self.client = pymongo.MongoClient(self.connection_string)
                print("MongoDB connected")
        except (Exception,) as e:
            print("Error in connecting MongoDB Cloud")

    def get_collection(self):
        if self.client is not None:
            database = self.client[self.database_name]
            collection = database.kidney_disease
            return collection
        else:
            print("Cannot connect with database due to failure in Internet connection")
            return None

    def read_data_from_mongodb(self):
        collection = self.get_collection()
        if collection is not None:
            data = pd.DataFrame.from_records(collection.find())
            return data
        else:
            print("Failed to read data from MongoDB Database. Check the internet connectivity")
            return None
