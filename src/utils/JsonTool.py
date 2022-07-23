def dict_to_list(src):
    des = []
    for _, v in src.items():
        des.append(v)
    return des


def list_to_dict(src):
    des = {}
    for i in range(len(src)):
        des[i] = src[i]
    return des
