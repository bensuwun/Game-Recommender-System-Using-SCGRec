import logging

datefmt = '%Y-%m-%d %H:%M:%S'
default_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt=datefmt)

def GetDefaultConsoleHandler():
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(default_format)
    return console_handler

def GetDefaultFileHandler():
    file_handler = logging.FileHandler('app.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(default_format)
    return file_handler