model = getModel(image_size=target, num_classes=1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=1, min_lr=1e-5)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='val-best.keras',
    monitor='val_loss',
    mode='min',
    save_best_only=True
)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy',
              metrics=['binary_accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
