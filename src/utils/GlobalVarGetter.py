class GlobalVarGetter:
    global_var = {}

    @staticmethod
    def set(global_var):
        GlobalVarGetter.global_var = global_var
        return GlobalVarGetter.global_var

    @staticmethod
    def get():
        return GlobalVarGetter.global_var
