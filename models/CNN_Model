# Build CNN model with ConvNeXt Tiny
def getModel(image_size, num_classes):
    model_input = tf.keras.Input(shape=(image_size, image_size, 3))
    transfer = tf.keras.applications.ConvNeXtTiny(
        weights='imagenet', include_preprocessing=True, include_top=False, input_tensor=model_input
    )
    x = transfer.output
    x = layers.GlobalMaxPool2D()(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-1))(x)
    model_output = layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model(inputs=model_input, outputs=model_output)
