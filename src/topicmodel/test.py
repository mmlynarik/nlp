import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices({"a": [1, 2], "b": [3, 4]})
for item in dataset:
    print(item, item["a"], item["b"])
