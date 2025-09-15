# Continue with existing imports
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
import numpy as np

# Extract features from the CNN model
def extract_cnn_features(base_model):
    """Extract intermediate features from ConvNeXtTiny"""
    selected_layers = ['convnext_tiny/stage_0/block_0/layer_scale_2',
                      'convnext_tiny/stage_1/block_0/layer_scale_2']
    return Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(name).output for name in selected_layers]
    )

class GraphCreationLayer(layers.Layer):
    def __init__(self, k_neighbors=8, **kwargs):
        super().__init__(**kwargs)
        self.k_neighbors = k_neighbors

    def call(self, features):
        # Get spatial dimensions
        height, width = features.shape[1:3]
        channels = features.shape[-1]

        # Reshape features to [batch, nodes, channels]
        node_features = tf.reshape(features, [-1, height * width, channels])

        # Create fixed spatial edge patterns
        indices = []
        for i in range(height):
            for j in range(width):
                current = i * width + j
                if j < width - 1:  # right
                    indices.append([current, current + 1])
                if i < height - 1:  # down
                    indices.append([current, current + width])
                if i < height - 1 and j < width - 1:  # diagonal
                    indices.append([current, current + width + 1])

        edge_indices = tf.constant(indices, dtype=tf.int32)
        return node_features, edge_indices

class GNNLayer(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        node_features_shape, _ = input_shape
        self.transform = layers.Dense(self.units, activation='relu')
        self.attention = layers.Dense(1, activation='tanh')
        self.combine = layers.Dense(self.units, activation='relu')
        super().build(input_shape)

    def call(self, inputs):
        node_features, edge_indices = inputs
        batch_size = tf.shape(node_features)[0]
        num_nodes = tf.shape(node_features)[1]

        # Transform node features
        node_features = self.transform(node_features)

        # Prepare indices for gathering from batched tensor
        batch_range = tf.range(batch_size)
        batch_range = tf.expand_dims(batch_range, -1)  # [batch_size, 1]
        batch_range = tf.tile(batch_range, [1, tf.shape(edge_indices)[0]])  # [batch_size, num_edges]
        batch_range = tf.reshape(batch_range, [-1])  # [batch_size * num_edges]

        # Repeat edge indices for each batch
        edge_indices_repeated = tf.tile(tf.expand_dims(edge_indices, 0), [batch_size, 1, 1])
        edge_indices_flat = tf.reshape(edge_indices_repeated, [-1, 2])

        # Gather source and target nodes
        gather_indices = tf.stack([batch_range, tf.reshape(edge_indices_flat[:, 0], [-1])], axis=1)
        source_nodes = tf.gather_nd(node_features, gather_indices)

        gather_indices = tf.stack([batch_range, tf.reshape(edge_indices_flat[:, 1], [-1])], axis=1)
        target_nodes = tf.gather_nd(node_features, gather_indices)

        # Compute attention scores
        edge_features = tf.concat([source_nodes, target_nodes], axis=-1)
        attention_weights = tf.nn.softmax(self.attention(edge_features), axis=0)

        # Weight the target nodes
        weighted_messages = attention_weights * target_nodes

        # Reshape for scattering
        weighted_messages = tf.reshape(weighted_messages, [batch_size, -1, self.units])

        # Aggregate messages using scatter_nd
        scattered_shape = tf.stack([batch_size, num_nodes, self.units])
        scattered_messages = tf.zeros(scattered_shape)

        # Create scatter indices
        scatter_indices = tf.stack([
            tf.repeat(tf.range(batch_size), tf.shape(edge_indices)[0]),
            tf.reshape(edge_indices_flat[:, 0], [-1])
        ], axis=1)

        scattered_messages = tf.tensor_scatter_nd_add(
            scattered_messages,
            scatter_indices,
            tf.reshape(weighted_messages, [-1, self.units])
        )

        # Combine with original features
        output = self.combine(tf.concat([node_features, scattered_messages], axis=-1))
        return output

def getModelWithGNN(image_size, num_classes):
    # Input layer
    model_input = tf.keras.Input(shape=(image_size, image_size, 3))

    # Base CNN
    base_cnn = tf.keras.applications.ConvNeXtTiny(
        weights='imagenet',
        include_preprocessing=True,
        include_top=False,
        input_tensor=model_input
    )

    # Extract CNN features
    cnn_features = base_cnn.output

    # Convert to graph structure
    node_features, edge_indices = GraphCreationLayer()(cnn_features)

    # Apply GNN layers
    x = node_features
    for units in [256, 128, 64]:
        x = GNNLayer(units)([x, edge_indices])

    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Dense layers
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-1))(x)

    # Output heads
    classification_output = layers.Dense(1, activation='sigmoid', name='classification')(x)
    ungradability_output = layers.Dense(1, activation='linear', name='ungradability')(x)

    return Model(inputs=model_input, outputs=[classification_output, ungradability_output])

# Create and compile the model
model = getModelWithGNN(image_size=target, num_classes=1)

# Compile with multiple outputs
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={
        'classification': 'binary_crossentropy',
        'ungradability': 'mse'
    },
    loss_weights={
        'classification': 1.0,
        'ungradability': 0.5
    },
    metrics={
        'classification': [
            'binary_accuracy',
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ],
        'ungradability': ['mae']
    }
)

# Modified data generator (unchanged from previous version)
def enhanced_generator(base_generator, include_ungradability=True):
    while True:
        batch_x, batch_y = next(base_generator)
        if include_ungradability:
            ungradability = np.random.uniform(0, 1, size=(batch_x.shape[0], 1))
            yield batch_x, {'classification': batch_y, 'ungradability': ungradability}
        else:
            yield batch_x, {'classification': batch_y, 'ungradability': np.zeros((batch_x.shape[0], 1))}

# Setup generators
train_gen_enhanced = enhanced_generator(trainGen)
val_gen_enhanced = enhanced_generator(valGen, include_ungradability=False)
test_gen_enhanced = enhanced_generator(testGen, include_ungradability=False)

# Train the model
history = model.fit(
    train_gen_enhanced,
    steps_per_epoch=len(trainGen),
    validation_data=val_gen_enhanced,
    validation_steps=len(valGen),
    epochs=20,
    callbacks=[reduce_lr, model_checkpoint_callback]
)
