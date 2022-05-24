def find_F_by_string(s):
    torch_module = __import__("torch")
    nn_module = getattr(torch_module, "nn")
    functional_module = getattr(nn_module, "functional")
    ans = getattr(functional_module, s)
    return ans


def find_class_by_string(module_name, file_name, class_name):
    find_module = __import__(module_name)
    find_file = getattr(find_module, file_name)
    ans = getattr(find_file, class_name)
    return ans


def find_opti_by_string(s):
    torch_module = __import__("torch")
    optim_module = getattr(torch_module, "optim")
    ans = getattr(optim_module, s)
    return ans
