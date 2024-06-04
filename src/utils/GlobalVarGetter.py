import sys
from collections import defaultdict


class GlobalVarGetter:
    global_var = defaultdict(dict)
    pos = -1

    @staticmethod
    def set(global_var):
        GlobalVarGetter.global_var = defaultdict(dict, global_var)
        if not isinstance(sys.argv, str):
            sys.argv.append({"config": global_var['config']})
            GlobalVarGetter.pos = len(sys.argv) - 1
        return GlobalVarGetter.global_var

    @staticmethod
    def get():
        if GlobalVarGetter.pos == -1:
            for index, i in enumerate(reversed(sys.argv)):
                if isinstance(i, defaultdict):
                    GlobalVarGetter.pos = index
            GlobalVarGetter.global_var = sys.argv[GlobalVarGetter.pos]
        return GlobalVarGetter.global_var
