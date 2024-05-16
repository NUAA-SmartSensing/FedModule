from collections import defaultdict


class GlobalVarGetter:
    global_var = defaultdict(dict)

    @staticmethod
    def set(global_var):
        GlobalVarGetter.global_var = defaultdict(dict, global_var)
        return GlobalVarGetter.global_var

    @staticmethod
    def get():
        return GlobalVarGetter.global_var
