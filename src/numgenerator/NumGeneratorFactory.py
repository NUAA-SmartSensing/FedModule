from numgenerator.StaticNumGenerator import StaticNumGenerator
from utils import ModuleFindTool


class NumGeneratorFactory:
    def __init__(self, config):
        self.config = config

    def create_num_generator(self, *args, **kwargs):
        if isinstance(self.config, int):
            return StaticNumGenerator(self.config)
        else:
            determineClientUpdateNumClass = ModuleFindTool.find_class_by_path(self.config["path"])
            return determineClientUpdateNumClass(self, self.config["params"], *args, **kwargs)

