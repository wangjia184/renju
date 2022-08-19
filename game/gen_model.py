# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in Tensorflow V2
Author : Jerry.Wang  vcer#qq.com
"""

from tabnanny import verbose
import tensorflow as tf

print( tf.__version__ )

BOARD_WIDTH = 15
BOARD_HEIGHT = 15

class Renju15x15Model(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(Renju15x15Model, self).__init__(*args, **kwargs)

        self.transposed_inputs = tf.keras.layers.Lambda( lambda x: tf.transpose(x, [0, 2, 3, 1]) )

        # 2. Common Networks Layers
        self.conv1 = tf.keras.layers.Conv2D( name="conv1",
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            data_format="channels_last",
            activation=tf.keras.activations.relu)

        self.conv2 = tf.keras.layers.Conv2D( name="conv2", 
            filters=64, 
            kernel_size=(3, 3), 
            padding="same", 
            data_format="channels_last", 
            activation=tf.keras.activations.relu)

        self.conv3 = tf.keras.layers.Conv2D( name="conv3",
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            data_format="channels_last",
            activation=tf.keras.activations.relu)

        # 3-1 Action Networks
        self.action_conv = tf.keras.layers.Conv2D( name="action_conv",
            filters=4,
            kernel_size=(1, 1),
            padding="same",
            data_format="channels_last",
            activation=tf.keras.activations.relu)

        # flatten tensor
        self.action_conv_flat = tf.keras.layers.Reshape( (-1, 4 * BOARD_HEIGHT * BOARD_WIDTH), name="action_conv_flat" )

        # 3-2 Full connected layer, the output is the log probability of moves
        # on each slot on the board
        self.action_fc = tf.keras.layers.Dense( BOARD_WIDTH * BOARD_HEIGHT,
            activation=tf.nn.log_softmax,
            name="action_fc")

        # 4 Evaluation Networks
        self.evaluation_conv = tf.keras.layers.Conv2D( name="evaluation_conv",
            filters=2,
            kernel_size=(1, 1),
            padding="same",
            data_format="channels_last",
            activation=tf.keras.activations.relu)

        self.evaluation_conv_flat = tf.keras.layers.Reshape( (-1, 2 * BOARD_HEIGHT * BOARD_WIDTH), name="evaluation_conv_flat" )

        self.evaluation_fc1 = tf.keras.layers.Dense( 64,
            activation=tf.keras.activations.relu,
            name="evaluation_fc1")

        self.evaluation_fc2 = tf.keras.layers.Dense( 1, 
            activation=tf.keras.activations.tanh,
            name="evaluation_fc2")

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.build((None, 4, BOARD_HEIGHT, BOARD_WIDTH))

    #Predict
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 4, BOARD_HEIGHT, BOARD_WIDTH], dtype=tf.float32, name='inputs')])
    def call(self, inputs):
        x = self.transposed_inputs(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        y1 = self.action_conv(x)
        y1 = self.action_conv_flat(y1)
        y1 = self.action_fc(y1)

        y2 = self.evaluation_conv(x)
        y2 = self.evaluation_conv_flat(y2)
        y2 = self.evaluation_fc1(y2)
        y2 = self.evaluation_fc2(y2)
        return (y1, y2)
    #Train function called from Rust which uses the keras model innate train_step function

    @tf.function
    def train_step(self, data):
        state_batch, labels = data
        mcts_probs, winner_batch = labels
        loss, entropy = self.raw_train_step(state_batch, mcts_probs, winner_batch)
        return { "loss" : loss, "entropy" : entropy }

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 4, BOARD_HEIGHT, BOARD_WIDTH],  dtype=tf.float32, name="state_batch"), 
                                  tf.TensorSpec(shape=[None, BOARD_HEIGHT * BOARD_WIDTH],  dtype=tf.float32, name="mcts_probs"),
                                  tf.TensorSpec(shape=[],  dtype=tf.float32, name="winner_batch") ])
    def raw_train_step(self, state_batch, mcts_probs, winner_batch):
        l2_penalty_beta = 1e-4

        trainable_vars = self.trainable_variables
        with tf.GradientTape() as tape:
            log_act_probs, pred_value = self(state_batch, training=True)  # Forward pass

            # Compute our own loss
            value_loss = tf.losses.mean_squared_error(winner_batch, pred_value)
            policy_loss = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.multiply(mcts_probs, log_act_probs[0]), 1)))
            l2_penalty = l2_penalty_beta * tf.add_n(
                [tf.nn.l2_loss(v) for v in trainable_vars if 'bias' not in v.name.lower()])
            loss = policy_loss + value_loss + l2_penalty

        # Compute gradients
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)

        entropy = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.exp(log_act_probs[0]) * log_act_probs[0], 1)))


        return (self.loss_tracker.result(), entropy)

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string, name="checkpoint_filename")])
    def export(self, checkpoint_path):
        tensor_names = [weight.name for weight in self.trainable_variables]
        tensors_to_save = [weight.read_value() for weight in self.trainable_variables]
        tf.raw_ops.Save(
            filename=checkpoint_path, tensor_names=tensor_names,
            data=tensors_to_save, name='save')
        return checkpoint_path 

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string, name="checkpoint_filename")])
    def restore(self, checkpoint_path):
        restored_tensors = {}
        for var in self.trainable_variables:
            restored = tf.raw_ops.Restore(
                file_pattern=checkpoint_path, tensor_name=var.name, dt=var.dtype,
                name='restore')
            var.assign(restored)
            restored_tensors[var.name] = restored
        #tf.print(restored_tensors)
        return checkpoint_path


    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker]

model = Renju15x15Model()
model.summary()

#print(tf.autograph.to_code(custom_loss))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3))



#Saving the model, explictly adding the concrete functions as signatures
model.save('renju_15x15_model', 
        save_format='tf', 
        signatures={
            'predict': model.call.get_concrete_function(), 
            'train' : model.raw_train_step.get_concrete_function(), 
            'export' : model.export.get_concrete_function(),
            'restore' : model.restore.get_concrete_function() })
