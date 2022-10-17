from configparser import ConfigParser
from cryptography.fernet import Fernet


class SecureConnect:
    def __init__(self, password=None, username=None):
        config = ConfigParser()
        config.read('config.ini')
        self.password = password
        self.username = username
        self._key = config['mongo']['key']
        self._user_name = config['mongo']['user_name']
        self._encrypted_password = config['mongo']['encrypted_password']

    def decrypt(self):
        key = Fernet(bytes(self._key,'utf-8'))
        myPass = (key.decrypt(self._encrypted_password))
        return str(myPass).replace("b'","").replace("'","")

    def get_user_name(self):
        return self._user_name

    def is_connection_success(self):
        if (self.password == self.decrypt()) and (self.username == self._user_name):
            return True
        else:
            return False
