import json


def getConfig(filename):
    with open(filename, 'r', encoding='utf8') as fp:
        config = json.load(fp)
    return config
