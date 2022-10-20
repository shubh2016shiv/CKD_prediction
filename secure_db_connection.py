from configparser import ConfigParser
from cryptography.fernet import Fernet


class SecureConnect:
    def __init__(self, password=None, username=None):
        config = ConfigParser()
        config.read('config.ini') # to read username, encrypted password and key for decryption
        self.password = password
        self.username = username
        self._key = config['mongo']['key'] # private variable to hold key
        self._user_name = config['mongo']['user_name'] # private variable to hold username
        self._encrypted_password = config['mongo']['encrypted_password'] # private variable to hold encrypted password
    
    def decrypt(self):
        """
        Decryption of encrypted password using key 
        :return: Decrypted password
        """
        key = Fernet(bytes(self._key, 'utf-8'))
        myPass = (key.decrypt(bytes(self._encrypted_password,'utf-8')))
        return str(myPass).replace("b'", "").replace("'", "")

    def get_user_name(self):
        # accessing the private variable for username
        return self._user_name

    def is_connection_valid(self)->bool:
        """
        Check if the connection is valid or not
        :return: bool
        """
        if (self.password == self.decrypt()) and (self.username == self._user_name):
            return True
        else:
            return False
