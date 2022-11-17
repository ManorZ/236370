import numpy as np


def sigmoid(x):
    """
    Parameters
    ----------
    x : np.array input data

    Returns
    -------
    np.array
        sigmoid of the input x

    """
    return 1 / (1 + np.exp(-x))
    raise NotImplementedError("To be implemented")


def sigmoid_prime(x):
    """
    Parameters
    ----------
    x : np.array input data

    Returns
    -------
    np.array
        derivative of sigmoid of the input x

    """
    return sigmoid(x) * (1 - sigmoid(x))
    raise NotImplementedError("To be implemented")


def random_weights(sizes):
    """
    Parameters
    ----------
    sizes : list of sizes

    Returns
    -------
    list
        list of xavier initialized np arrays weight matrices

    """
    return [xavier_initialization(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]
    raise NotImplementedError("To be implemented")


def zeros_weights(sizes):
    """
    Parameters
    ----------
    sizes : list of sizes

    Returns
    -------
    list
        list of zero np arrays weight matrices

    """
    return [np.zeros((sizes[i], sizes[i+1])) for i in range(len(sizes)-1)]
    raise NotImplementedError("To be implemented")


def zeros_biases(list):
    """
    Parameters
    ----------
    sizes : list of sizes

    Returns
    -------
    list
        list of zero np arrays bias matrices

    """
    return [np.zeros(list[i]) for i in range(len(list))]
    raise NotImplementedError("To be implemented")


def create_batches(data, labels, batch_size):
    """
    Parameters
    ----------
    data : np.array of input data
    labels : np.array of input labels
    batch_size : int size of batch

    Returns
    -------
    list
        list of tuples of (data batch of batch_size, labels batch of batch_size)

    """
    return \
        [ x for x in
        [(data[batch_size * i : batch_size * (i + 1)], labels[batch_size * i : batch_size * (i + 1)]) for i in range(len(data)//batch_size)] + \
        [(data[batch_size * (len(data)//batch_size) :], labels[batch_size * (len(data)//batch_size) :])] if x[0].shape[0] != 0
        ]
    raise NotImplementedError("To be implemented")


def add_elementwise(list1, list2):
    """
    Parameters
    ----------
    list1 : np.array of numbers
    list2 : np.array of numbers

    Returns
    -------
    list
        list of sum of each two elements by index
    """
    return [l1 + l2 for l1, l2 in zip(list1, list2)]
    raise NotImplementedError("To be implemented")

def xavier_initialization(m, n):
    xavier = 1 / (m ** 0.5)
    return np.random.uniform(low=-xavier, high=xavier, size=(m, n))


if __name__ == '__main__':
    sizes = [4,6,8,6,4]
    xavier_weights = random_weights(sizes)
    zero_weights = zeros_weights(sizes)
    data = np.reshape(np.arange(200), (100,2))
    labels = np.arange(100)
    batch_size = 6
    batch_list = create_batches(data, labels, batch_size)
    print('Bye Bye')
