class GlobalVarGetter:
    __instance = None

    def __new__(cls):
        if not cls.__instance:
            cls.__instance = super(GlobalVarGetter, cls).__new__(cls)
            cls.__instance.__global_var = {}
        return cls.__instance
    
    def set(self, global_var):
        self.__global_var = global_var
        return self.__global_var
        
    def get(self):
        return self.__global_var
