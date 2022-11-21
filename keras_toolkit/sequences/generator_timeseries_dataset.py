import numpy as np
import tensorflow as tf


def timeseries_dataset_multistep(
    features, labels, input_sequence_length, output_sequence_length, batch_size
):
    """_summary_

    Args:
        features (_type_): _description_
        labels (_type_): _description_
        input_sequence_length (_type_): _description_
        output_sequence_length (_type_): _description_
        batch_size (_type_): _description_
    """

    def extract_output(lst_output):
        return lst_output[:output_sequence_length]

    feature_ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        features, None, input_sequence_length, batch_size=1
    ).unbatch()
    label_ds = (
        tf.keras.preprocessing.timeseries_dataset_from_array(
            labels, None, input_sequence_length, batch_size=1
        )
        .skip(input_sequence_length)
        .unbatch()
        .map(extract_output)
    )

    return tf.data.Dataset.zip((feature_ds, label_ds)).batch(batch_size)


def timeseries_dataset_one_step(
    features, labels, input_sequence_length, batch_size
):

    """_summary_

    Args:
        features (_type_): _description_
        labels (_type_): _description_
        input_sequence_length (_type_): _description_
        batch_size (_type_): _description_

    Returns:
        _type_: _description_
    """
    return tf.keras.preprocessing.timeseries_dataset_from_array(
        features[:-1],
        np.roll(labels, -input_sequence_length, axis=0)[:-1],
        input_sequence_length,
        batch_size=batch_size,
    )


def timeseries_dataset_multistep_combined(
    features, label_slice, input_sequence_length,
    output_sequence_length, batch_size
):

    """_summary_

    Args:
        features (_type_): _description_
        label_slice (_type_): _description_
        input_sequence_length (_type_): _description_
        output_sequence_length (_type_): _description_
        batch_size (_type_): _description_

    Returns:
        _type_: _description_
    """
    feature_ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        features,
        None,
        input_sequence_length + output_sequence_length,
        batch_size=batch_size,
    )

    def split_feature_label(x):
        return (
            x[:, :input_sequence_length, :],
            x[:, input_sequence_length:, label_slice],
        )

    feature_ds = feature_ds.map(split_feature_label)

    return feature_ds
