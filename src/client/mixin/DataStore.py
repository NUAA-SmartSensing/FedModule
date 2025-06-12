class DataStore:
    """
    统一管理所有Client的数据，支持自定义共享/隔离策略。
    - 共享策略：如模型参数等全局共享
    - 隔离策略：如client id、BN参数、特征数据等每个client独立
    - 支持为特定key注册自定义存储策略（如模型参数的特殊存储方式）
    """
    def __init__(self, share_keys=None, isolate_keys=None, shared_dict=None):
        # share_keys: 需要全局共享的key列表
        # isolate_keys: 需要隔离的key列表
        # 其余key默认隔离
        self.share_keys = set(share_keys) if share_keys else set()
        self.isolate_keys = set(isolate_keys) if isolate_keys else set()
        self._shared = shared_dict if shared_dict is not None else {}
        self._isolated = {}
        self._strategies = {}  # key: str -> {'get': func, 'set': func}

    def register_strategy(self, key, get_func=None, set_func=None):
        """
        为特定key注册自定义get/set方法。
        get_func(client_id, key, default) -> value
        set_func(client_id, key, value) -> None
        """
        self._strategies[key] = {'get': get_func, 'set': set_func}

    def unregister_strategy(self, key):
        if key in self._strategies:
            del self._strategies[key]

    def get(self, client_id, key, default=None):
        if key in self._strategies and self._strategies[key]['get']:
            return self._strategies[key]['get'](client_id, key, default)
        if key in self.share_keys:
            return self._shared.get(key, default)
        else:
            return self._isolated.get(client_id, {}).get(key, default)

    def set(self, client_id, key, value):
        if key in self._strategies and self._strategies[key]['set']:
            return self._strategies[key]['set'](client_id, key, value)
        if key in self.share_keys:
            self._shared[key] = value
        else:
            if client_id not in self._isolated:
                self._isolated[client_id] = {}
            self._isolated[client_id][key] = value

    def get_all(self, client_id=None):
        # 返回该client所有数据（合并共享和隔离）
        result = dict(self._shared)
        if client_id is not None and client_id in self._isolated:
            result.update(self._isolated[client_id])
        return result

    def set_all(self, client_id, data_dict):
        for k, v in data_dict.items():
            self.set(client_id, k, v)

    # 可扩展：支持hook、跨client共享等
