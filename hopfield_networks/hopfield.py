import numpy as np
from Pixart import Pixart

def sign(a):
    if a > 0:
        return 1.0
    else:
        return -1.0


def load_memories(path='../pixart/'):
    filenames = ['6', '4', '7']
    memories = []
    for filename in filenames:
        matrix = np.array(np.load(path + filename + '.npy').reshape(-1,1), dtype=float)
        print(matrix.reshape((7,4)))
        matrix[matrix == 0] = -1
        memories.append(matrix)
    return memories


def evolve(W, test):
    oldtest = test
    vsign = np.vectorize(sign)
    index = 0
    while True:
        # pixart = Pixart(pixelResolution=(10, 10), matrix=oldtest.reshape((10,10)))
        # pixart.run()

        # print(index)
        index += 1
        newtest = vsign(W @ oldtest)

        if np.array_equal(oldtest, newtest) or index == 2000:
            break

        oldtest = np.copy(newtest)
    return newtest


if __name__ == "__main__":
    memories = load_memories()
    N = memories[0].shape[0]
    W = np.zeros((N, N))
    for memory in memories:
        W += (1 / N) * memory @ memory.T
    for i in range(N):
        W[i,i] = 0

    pixart = Pixart(pixelResolution=(7, 4), windowSize=(400, 700))
    pixart.run()
    matrix = np.array(pixart.getMatrix(), dtype=float)

    # matrix = np.array(np.load('../pixart/6.npy').reshape(-1, 1), dtype=float)
    matrix[matrix == 0] = -1
    print(matrix)
    result = evolve(W, matrix.reshape((-1,1))).reshape(matrix.shape)
    print(result)
    pixart = Pixart(pixelResolution=(7, 4), matrix=result, windowSize=(400, 700))
    pixart.run()

