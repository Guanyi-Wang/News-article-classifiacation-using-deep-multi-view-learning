import tensorflow as tf
def print_all_variables(train_only=True):
    if train_only:
        t_vars = tf.trainable_variables()
        print("printing trainable variables:")
    else:
        t_vars = tf.global_variables()
        print("printing global variables")
    for idx, v in enumerate(t_vars):
        print("index:{};shape:{} name:{}".format(idx, str(v.get_shape()), v.name))