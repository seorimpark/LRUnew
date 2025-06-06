import jax.numpy as jnp
import numpy as np
import tensorflow as tf


def vec_bin_array(
    arr, m
):  # https://stackoverflow.com/questions/22227595/convert-integer-to-binary-array-with-suitable-padding
    """
    Arguments:
    arr: Numpy array of positive integers
    m: Number of bits of each integer to retain

    Returns a copy of arr with every element replaced with a bit vector.
    Bits encoded as int8's.12
    """

    to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(m))
    strs = to_str_func(arr)
    ret = np.zeros(list(arr.shape) + [m], dtype=np.int8)
    for bit_ix in range(0, m):
        fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == "1")
        ret[..., bit_ix] = fetch_bit_func(strs).astype("int8")

    return ret


# Import data


def data_transformation(dataset_name, trans_to_8bit):
    if dataset_name == "MNIST":
        dataset = tf.keras.datasets.mnist.load_data()
        train = dataset[0]
        test = dataset[1]

        train_x_seq = train[0].shape[0]
        train_x_len = int(jnp.prod(jnp.array(train[0].shape[1:])))
        test_x_seq = test[0].shape[0]
        test_x_len = int(jnp.prod(jnp.array(test[0].shape[1:])))
        if trans_to_8bit:
            train_x_size = 8
            test_x_size = 8

            train_x = vec_bin_array(train[0], train_x_size)
            train_x = train_x.reshape((train_x_seq, train_x_len, train_x_size))

            train_y = train[1].reshape(train_x_seq)

            train_y_class = len(jnp.unique(train_y))

            test_x = vec_bin_array(test[0], test_x_size)
            test_x = test_x.reshape((test_x_seq, test_x_len, test_x_size))

            test_y = test[1].reshape(test_x_seq)

        else:
            train_x_size = 1
            test_x_size = 1
            train_x = train[0].reshape((train_x_seq, train_x_len, train_x_size)) / 255
            train_y = train[1].reshape(train_x_seq)
            train_y_class = len(jnp.unique(train_y))
            test_x = test[0].reshape((test_x_seq, test_x_len, test_x_size)) / 255
            test_y = test[1].reshape(test_x_seq)

    if dataset_name == "CIFAR10":
        dataset = tf.keras.datasets.cifar10.load_data()
        train = dataset[0]
        test = dataset[1]

        train_x_seq = train[0].shape[0]
        train_x_len = int(jnp.prod(jnp.array(train[0].shape[1:-1])))
        train_x_size = int(jnp.prod(jnp.array(train[0].shape[-1])))

        test_x_seq = test[0].shape[0]
        test_x_len = int(jnp.prod(jnp.array(test[0].shape[1:-1])))
        test_x_size = int(jnp.prod(jnp.array(train[0].shape[-1])))

        if trans_to_8bit:
            train_x_size = 24
            test_x_size = 24
            train_x = vec_bin_array(train[0], 8)
            train_x = train_x.reshape((train_x_seq, train_x_len, test_x_size))

            train_y = train[1].reshape(train_x_seq)
            train_y_class = len(jnp.unique(train_y))

            test_x = vec_bin_array(test[0], 8)
            test_x = test_x.reshape((test_x_seq, test_x_len, test_x_size))

            test_y = test[1].reshape(test_x_seq)

        else:
            train_x = train[0].reshape((train_x_seq, train_x_len, train_x_size)) / 255
            train_y = train[1].reshape(train_x_seq)
            train_y_class = len(jnp.unique(train_y))
            test_x = test[0].reshape((test_x_seq, test_x_len, test_x_size)) / 255
            test_y = test[1].reshape(test_x_seq)
            train_x_size = int(jnp.prod(jnp.array(train[0].shape[-1])))
            test_x_size = int(jnp.prod(jnp.array(train[0].shape[-1])))
    return (
        train,
        test,
        train_x,
        train_x_size,
        train_x_len,
        test_x,
        test_x_len,
        test_x_seq,
        train_y,
        train_y_class,
        test_y,
    )


def group_tuples_to_nested_dict(params):
    nested_dict = {}
    for outer_key, inner_key in params:
        if outer_key not in nested_dict:
            nested_dict[outer_key] = {}
        nested_dict[outer_key][inner_key] = inner_key
    return nested_dict
