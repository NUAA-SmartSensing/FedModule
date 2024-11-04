import importlib


def find_class_by_path(path: str):
    path_list = path.split(".")
    for i in reversed(list(range(len(path_list)))):
        entry = '.'.join(path_list[:i+1])
        try:
            module = importlib.import_module(entry)
        except ModuleNotFoundError:
            continue
        attr_list = path_list[i:] if i > 0 else path_list[i+1:]
        try:
            for package in attr_list:
                module = getattr(module, package)
            return module
        except AttributeError:
            pass
    raise Exception(f"Module {path} not found.")


def generate_object_by_path(path: str, params: dict, else_params=None):
    target_class = find_class_by_path(path)
    if else_params is not None:
        params = params.update(else_params)
    target_object = target_class(**params)
    return target_object


if __name__ == '__main__':
    print(find_class_by_path("client.ActiveClient.ActiveClient"))
