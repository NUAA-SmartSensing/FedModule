import json


def getConfig():
    with open('../../config.json', 'r', encoding='utf8')as fp:
        config = json.load(fp)
    return config
