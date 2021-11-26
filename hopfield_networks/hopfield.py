import numpy as np
from Pixart import Pixart
from typing import List


def sign(a: float) -> float:
    """
    SUMMARY
        The mathematical sign function. Note that 0 is mapped differently.
    PARAMETERS
        a float: the input number
    RETURN
        float64: either +1.0 or -1.0
    """
    if a > 0:
        return 1.0
    else:
        return -1.0


def load_memories(path: str = 'digits/') -> List[np.ndarray]:
    """
    SUMMARY
        Loads memories from files. These contain the digits 6, 4 and 7.
    PARAMETERS
        path str: the folder where the files of the digits reside
    RETURN
        List[np.ndarray]: list of memories
    """
    filenames = ['6', '4', '7']
    memories = []
    for filename in filenames:
        matrix = np.array(np.load(path + filename + '.npy').reshape(-1,1), dtype=float)
        print(matrix.reshape((7,4)))
        matrix[matrix == 0] = -1
        memories.append(matrix)
    return memories


def evolve(W: np.ndarray, test: np.ndarray) -> np.ndarray:
    """
    SUMMARY
        Lets the input converge.
    PARAMETERS
        W np.ndarray: the weights of the network
        test np.ndarray: the test from the user input
    RETURN
        np.ndarray: the "memory" the model converged to
    """
    oldtest = test
    vsign = np.vectorize(sign)
    index = 0
    while True:
        index += 1
        newtest = vsign(W @ oldtest)

        if np.array_equal(oldtest, newtest) or index == 2000:
            break

        oldtest = np.copy(newtest)
    return newtest


# opens the pixart interface for user input and then runs the model
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

    matrix[matrix == 0] = -1
    print(matrix)
    result = evolve(W, matrix.reshape((-1,1))).reshape(matrix.shape)
    print(result)
    pixart = Pixart(pixelResolution=(7, 4), matrix=result, windowSize=(400, 700))
    pixart.run()

