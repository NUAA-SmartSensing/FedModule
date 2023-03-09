def find_class_by_path(path: str):
    path_list = path.split(".")
    module = __import__(path_list[0])
    path_list = path_list[1:]
    for package in path_list:
        module = getattr(module, package)
    return module


def generate_object_by_path(path: str, params: dict, else_params=None):
    target_class = find_class_by_path(path)
    if else_params is not None:
        params = params.update(else_params)
    target_object = target_class(**params)
    return target_object
