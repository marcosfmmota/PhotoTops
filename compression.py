import numpy as np


def write_as_mcf(ndarray, filename):

    file = open(filename + ".mcf", "wb")
    file.write(ndarray.tobytes())


def read_as_np(filename: np.ndarray):

    nparray = np.fromfile(filename + ".mcf", dtype=np.uint8)
    return nparray


def lzw_compress(image: np.ndarray) -> np.ndarray:

    di_limit = 512
    di = [[i] for i in range(256)]

    w = []
    result = []

    for p in image:
        wc = w + [p]
        if wc in di:
            print(wc)
            w = wc
        else:
            result.append(di.index(w))
            if len(di) < di_limit:
                di += [w]
                print(w)
                w += [p]

    if w:
        result.append(di.index(w))

    return np.array(result)


def lzw_decompress(vec: np.ndarray):

    compressed: list = vec.tolist()
    di = [[i] for i in range(256)]
    w = compressed.pop()
    result = [w]
    for k in compressed:
        print(k)
        if [k] in di:
            entry = di[k]
        elif k == len(di):
            entry = [k] + di[0]
        else:
            raise ValueError("It is not possible to decompress the file")
        result += entry

        di += [[w] + [entry[0]]]
        w = entry

    return np.array(result)


def run_length(vec: np.ndarray):


    run = []
    i = 0
    while i < vec.size:
        count = 0
        for j in range(i, vec.size):
            if vec[i] == vec[j]:
                count += 1
            else:
                break

        run += [count, vec[i]]
        i += count

    return np.array(run)


def run_length_inv(run: np.ndarray):

    vec = []

    for i in range(0, run.size, 2):
        for j in range(run[i]):
            vec += [run[i+1]]

    return np.array(vec)


def compress(image):

    shape = image.shape
    img_vect = image.reshape(image.size)

    r = run_length(img_vect)

    print(r.shape)
    return r


def decompress(vec, shape):

    dr = run_length_inv(vec)

    img = dr.reshape(shape)

    return img
