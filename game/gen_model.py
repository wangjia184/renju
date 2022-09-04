# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in Tensorflow V2
Author : Jerry.Wang  vcer#qq.com
"""

import tensorflow as tf
import tensorflow_probability as tfp

print( tf.__version__ )


def create_model(board_width, board_height):

    class RenjuModel(tf.Module):
        def __init__(self):
            l2_penalty_beta = 1e-4

            # Define the tensorflow neural network
            # 1. Input:
            self.inputs = tf.keras.Input( shape=(4, board_height, board_width), dtype=tf.dtypes.float32, name="input")
            self.transposed_inputs = tf.keras.layers.Lambda( lambda x: tf.transpose(x, [0, 2, 3, 1]) )(self.inputs)

            # 2. Common Networks Layers
            self.conv1 = tf.keras.layers.Conv2D( name="conv1",
                filters=32,
                kernel_size=(3, 3),
                padding="same",
                data_format="channels_last",
                activation=tf.keras.activations.relu,
                kernel_regularizer=tf.keras.regularizers.L2(l2_penalty_beta)
                )(self.transposed_inputs)

            self.conv2 = tf.keras.layers.Conv2D( name="conv2", 
                filters=64, 
                kernel_size=(3, 3), 
                padding="same", 
                data_format="channels_last", 
                activation=tf.keras.activations.relu,
                kernel_regularizer=tf.keras.regularizers.L2(l2_penalty_beta)
                )(self.conv1)

            self.conv3 = tf.keras.layers.Conv2D( name="conv3",
                filters=128,
                kernel_size=(3, 3),
                padding="same",
                data_format="channels_last",
                activation=tf.keras.activations.relu,
                kernel_regularizer=tf.keras.regularizers.L2(l2_penalty_beta)
                )(self.conv2)

            # 3-1 Action Networks
            self.action_conv = tf.keras.layers.Conv2D( name="action_conv",
                filters=4,
                kernel_size=(1, 1),
                padding="same",
                data_format="channels_last",
                activation=tf.keras.activations.relu,
                kernel_regularizer=tf.keras.regularizers.L2(l2_penalty_beta)
                )(self.conv3)

            # flatten tensor
            self.action_conv_flat = tf.keras.layers.Reshape( (-1, 4 * board_height * board_width), name="action_conv_flat" 
            )(self.action_conv)

            # 3-2 Full connected layer, the output is the log probability of moves
            # on each slot on the board
            self.action_fc = tf.keras.layers.Dense( board_height * board_width,
                activation=tf.nn.log_softmax,
                name="action_fc",
                kernel_regularizer=tf.keras.regularizers.L2(l2_penalty_beta)
                )(self.action_conv_flat)

            # 4 Evaluation Networks
            self.evaluation_conv = tf.keras.layers.Conv2D( name="evaluation_conv",
                filters=2,
                kernel_size=(1, 1),
                padding="same",
                data_format="channels_last",
                activation=tf.keras.activations.relu,
                kernel_regularizer=tf.keras.regularizers.L2(l2_penalty_beta)
                )(self.conv3)

            self.evaluation_conv_flat = tf.keras.layers.Reshape( (-1, 2 * board_height * board_width),
                name="evaluation_conv_flat" 
                )(self.evaluation_conv)

            self.evaluation_fc1 = tf.keras.layers.Dense( 64,
                activation=tf.keras.activations.relu,
                name="evaluation_fc1",
                kernel_regularizer=tf.keras.regularizers.L2(l2_penalty_beta)
                )(self.evaluation_conv_flat)

            self.evaluation_fc2 = tf.keras.layers.Dense( 1, 
                activation=tf.keras.activations.tanh,
                name="evaluation_fc2",
                kernel_regularizer=tf.keras.regularizers.L2(l2_penalty_beta)
                )(self.evaluation_fc1)
                

            self.model = tf.keras.Model(inputs=self.inputs, outputs=[self.action_fc, self.evaluation_fc2], name="renju_model")
            self.model.summary()
 
            self.lr = tf.Variable(0.002, trainable=False, dtype=tf.dtypes.float32)

            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = self.lr),
                    loss=[self.action_loss, tf.keras.losses.MeanSquaredError()],
                    metrics=['accuracy'])


        @tf.function(input_signature=[ tf.TensorSpec([None, 1, board_height * board_width], tf.float32),
            tf.TensorSpec([None, 1, board_height * board_width], tf.float32)
        ])
        def action_loss(self, labels, predictions):
            #tf.print(labels, summarize=-1)
            #tf.print(predictions, summarize=-1)
            # labels are probabilities; predictions are logits
            return tf.negative(tf.reduce_mean(
                        tf.reduce_sum(tf.multiply(labels, predictions), 2)))

 
           

        @tf.function(input_signature=[
            tf.TensorSpec([None, 4, board_height, board_width], tf.float32),
        ])
        def predict(self, state_batch):
            if tf.shape(state_batch)[0] > 1:
                tf.print(state_batch, summarize=-1)
            x = self.model(state_batch)
            if tf.shape(state_batch)[0] > 1:
                tf.print(x, summarize=-1)
            return x

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, 4, board_height, board_width],  dtype=tf.float32), 
                                  tf.TensorSpec(shape=[None, 1, board_height * board_width],  dtype=tf.float32),
                                  tf.TensorSpec(shape=[None, 1, 1],  dtype=tf.float32),
                                  tf.TensorSpec(shape=[1],  dtype=tf.float32) ])
        def train(self, state_batch, mcts_probs, winner_batch, lr):
            self.lr.assign(tf.gather(lr, 0))
            with tf.GradientTape() as tape:
                predictions = self.model(state_batch, training=True)  # Forward pass
                # the loss function is configured in `compile()`
                loss = self.model.compiled_loss([mcts_probs, winner_batch], predictions, regularization_losses=self.model.losses)
 
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))

            entropy = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.exp(predictions[0][0]) * predictions[0][0], 1)))

            return (loss, entropy)

        

        @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
        def save(self, checkpoint_path):
            tensor_names = [weight.name for weight in self.model.weights]
            tensors_to_save = [weight.read_value() for weight in self.model.weights]
            tf.raw_ops.Save(
                filename=checkpoint_path, tensor_names=tensor_names,
                data=tensors_to_save, name='save')
            return checkpoint_path

        @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
        def restore(self, checkpoint_path):
            restored_tensors = {}
            for var in self.model.weights:
                restored = tf.raw_ops.Restore( file_pattern=checkpoint_path, tensor_name=var.name, dt=var.dtype, name='restore')
                var.assign(restored)
                restored_tensors[var.name] = restored
            return checkpoint_path

        @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
        def random_choose_with_dirichlet_noice(self, probs):
            concentration = 0.3*tf.ones(tf.size(probs))
            dist = tfp.distributions.Dirichlet(concentration)
            p = 0.75*probs + 0.25*dist.sample(1)[0]
            samples = tf.random.categorical(tf.math.log([p]), 1)
            return samples[0] # selected index


    return RenjuModel()


model = create_model( 15, 15)

#Saving the model, explictly adding the concrete functions as signatures
model.model.save('renju_15x15_model', 
        save_format='tf', 
        signatures={
            'predict': model.predict.get_concrete_function(), 
            'train' : model.train.get_concrete_function(), 
            'save' : model.save.get_concrete_function(),
            'restore' : model.restore.get_concrete_function(),
            'random_choose_with_dirichlet_noice' : model.random_choose_with_dirichlet_noice.get_concrete_function() 
        })
