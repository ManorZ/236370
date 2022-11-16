import numpy as np


def lists_ex1(l):
    # Loop over the list l and return how many elements
    # Are greater than zero
    pass


def lists_ex2(l, k):
    # Sort the list in reverse order and return a list of the first k elements
    # Try using slicing
    pass


def list_comp_ex(l):
    # Return how many elements are greater than zero in a one liner
    pass


def dicts_ex(l):
    # Given a list of numbers l, return a dictionary in which
    # every number appearing in l is a key, and its value is the number
    # of times it appears. Try not to use python's Counter
    pass


def strings_ex(s):
    # Given a string s, return the number of upper-case characters
    pass


def numpy_ex1(arr):
    # Check that the array is two dimensional, and that it is a symmetric matrix
    # Remember that a matrix M is symmetric iff M == trnspose(M)
    # Return a boolean
    pass


def numpy_ex2(arr):
    # Increase by 1 only the numbers in the array that are positive
    # Do not use loops!
    pass


def numpy_ex3(arr, k):
    # Given a numpy array, return the position of the k largest numbers
    # Do not use loops!
    pass


def numpy_ex4(arr, vec):
    # Given a 2d array and a 1d vector, return the row in the array
    # that is closest to the vector (l2 distance).
    # Do not use loops! Try using broadcasting instead
    pass


def test():
    # lists_ex1
    assert lists_ex1([]) == 0
    assert lists_ex1([-1, -2, 0]) == 0
    assert lists_ex1([1.0, 2.3, 4.0, -1.0]) == 3
    print("lists_ex1: PASSED")

    # lists_ex2
    assert lists_ex2([1, 2, 3], 0) == []
    assert lists_ex2([1, 2, 3, 4], 1) == [4]
    assert lists_ex2([1, 2, 3, 4], 3) == [4, 3, 2]
    print("lists_ex2: PASSED")

    # list_comp_ex1
    assert list_comp_ex([]) == 0
    assert list_comp_ex([-1, -2, 0]) == 0
    assert list_comp_ex([1.0, 2.3, 4.0, -1.0]) == 3
    print("list_comp_ex1: PASSED")

    # dicts_ex
    counter = dicts_ex([1, 1, 2, 1, 1, 3, 4, 2, 1, 1])
    assert counter[1] == 6
    assert counter[2] == 2
    print("dicts_ex: PASSED")

    # strings_ex
    assert strings_ex("HellO") == 2
    assert strings_ex("123") == 0
    assert strings_ex("TEST") == 4
    print("strings_ex: PASSED")

    # numpy_ex1
    assert not numpy_ex1(np.asarray([1]))
    assert not numpy_ex1(np.reshape(np.arange(16), (4, 4)))
    assert numpy_ex1(np.eye(5, 5))
    print("numpy_ex1: PASSED")

    # numpy_ex2
    assert list(numpy_ex2(np.asarray([1, -4, 5, 2, -7]))) == [2, -4, 6, 3, -7]
    print("numpy_ex2: PASSED")

    # numpy_ex3
    assert list(numpy_ex3(np.asarray([1, -4, 5, 2, -7]), 1)) == [2]
    assert list(numpy_ex3(np.asarray([1, -4, 5, 2, -7]), 3)) == [2, 3, 0]
    print("numpy_ex3: PASSED")

    # numpy_ex4
    assert numpy_ex4(np.reshape(np.arange(16), (4, 4)), np.asarray([9, 9, 9, 9])) == 2
    print("numpy_ex4: PASSED")
	

if __name__ == '__main__':
    test()
