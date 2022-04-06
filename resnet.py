import tensorflow as tf

def create(x, num_outputs):
    '''
        args:
            x               network input
            num_outputs     number of logits
    '''
    is_training = tf.get_variable('is_training', (), dtype = tf.bool, trainable = False)

    # TODO #max_pool_layer = tf.nn.max_pool(relu_layer,3,2)

    conv_layer = tf.layers.conv2d(x, filters=64, kernel_size=7, strides=2)
    batch_norm_layer = tf.layers.batch_normalization(conv_layer, training=is_training)
    relu_layer = tf.nn.relu(batch_norm_layer)
    max_pool_layer = tf.layers.max_pooling2d(relu_layer, pool_size=3, strides=2)


    resnet_block1 = resnet_block(max_pool_layer, 64, 1, is_training)
    input_block2 = tf.add( max_pool_layer, resnet_block1 )


    resnet_block2 = resnet_block(input_block2, 128, 2, is_training)
    convolved_input2 = tf.layers.conv2d(input_block2, filters= 128, kernel_size=1 ,strides=2)
    batch_norm_input2 = tf.layers.batch_normalization(convolved_input2, training=is_training)
    input_block3 = tf.add(batch_norm_input2, resnet_block2)

    resnet_block3 = resnet_block(input_block3, 256, 2, is_training)
    convolved_input3 = tf.layers.conv2d(input_block3, filters=256, kernel_size=1 ,strides=2)
    batch_norm_input3 = tf.layers.batch_normalization(convolved_input3, training=is_training)
    input_block4 = tf.add(batch_norm_input3, resnet_block3)


    resnet_block4 = resnet_block(input_block4, 512, 2, is_training)
    convolved_input4 = tf.layers.conv2d(input_block4, filters=512, kernel_size=1, strides=2)
    batch_norm_input4 = tf.layers.batch_normalization(convolved_input4, training=is_training)
    block_sum = tf.add(batch_norm_input4, resnet_block4)

    gap_layer = tf.reduce_mean(block_sum, axis=[1, 2])
    #gap_layer = tf.keras.layers.GlobalAveragePooling2D()(block_sum)
    #flat = tf.layers.flatten(gap_layer)
    fc_layer = tf.layers.dense(gap_layer, units = num_outputs)

    return fc_layer


def resnet_block(x, channels, stride, is_training):

    block1_conv = tf.layers.conv2d(x, filters= channels, kernel_size=3, strides=stride, padding="same")
    block1_batch_norm = tf.layers.batch_normalization(block1_conv, training=is_training)
    block1_relu = tf.nn.relu(block1_batch_norm)

    block1b_conv = tf.layers.conv2d(block1_relu, filters= channels, kernel_size=3, padding="same")
    block1b_batch_norm = tf.layers.batch_normalization(block1b_conv , training=is_training)
    block1b_relu = tf.nn.relu(block1b_batch_norm)

    return block1b_relu



