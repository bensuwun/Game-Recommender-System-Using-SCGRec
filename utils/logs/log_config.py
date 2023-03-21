import logging

datefmt = '%Y-%m-%d %H:%M:%S'
default_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt=datefmt)

def SetDefaultConfig(logger):
    AddDefaultConsoleHandler(logger)
    AddDefaultFileHandler(logger)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

def AddDefaultConsoleHandler(logger):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(default_format)
    logger.addHandler(console_handler)

def AddDefaultFileHandler(logger):
    file_handler = logging.FileHandler('app.log')
    file_handler.setFormatter(default_format)
    logger.addHandler(file_handler)