from configparser import ConfigParser


class Config(ConfigParser):

    def __init__(self):
        super().__init__()
        self.file_name = ''

    def read(self, file_name):
        self.file_name = file_name
        return super().read(file_name)

    def write(self):
        with open(self.file_name, 'w') as f:
            super().write(f)


def load_config(file_name):
    config = Config()
    config.read(file_name)
    return config
