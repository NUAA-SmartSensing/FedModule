import random


def shuffle_random(l, r):
    arr = list(range(l, r))
    i = r - 1
    while i >= l:
        j = random.randint(l, i)
        arr[i], arr[j] = arr[j], arr[i]
        i = i - 1
    return arr
