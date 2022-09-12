# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in Tensorflow V2
Author : Jerry.Wang  vcer#qq.com
"""
import traceback
import pickle
import numpy as np
import platform
import tensorflow as tf
import tensorflow_probability as tfp

print( "platform: ", platform.platform() )
print( "tensorflow version:", tf.__version__ )
print( "tensorflow_probability version:", tfp.__version__ )


def create_model(board_width, board_height):

    class RenjuModel(tf.Module):
        def __init__(self):
            l2_penalty_beta = 1e-4

            # Define the tensorflow neural network
            # 1. Input:
            self.inputs = tf.keras.Input( shape=(4, board_height, board_width), dtype=tf.dtypes.float32, name="input")

            # convert from  NCHW(channels_first) to NHWC(channels_last) because channels_first is not supported by CPU
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
            self.action_conv_flat = tf.keras.layers.Flatten()(self.action_conv)

            # 3-2 Full connected layer, the output is the log probability of moves
            # on each slot on the board
            self.action_fc = tf.keras.layers.Dense( board_height * board_width,
                activation=tf.keras.activations.softmax,
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

            self.evaluation_conv_flat = tf.keras.layers.Flatten()(self.evaluation_conv)

            self.evaluation_fc1 = tf.keras.layers.Dense( 64,
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

            # loss = (z - v)^2 + pi^T * log(p) + c||theta||^2
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = self.lr),
                    loss=[tf.keras.losses.CategoricalCrossentropy(), tf.keras.losses.MeanSquaredError()])




         


        @tf.function(input_signature=[tf.TensorSpec(shape=[None, 4, board_height, board_width],  dtype=tf.float32), 
                                  tf.TensorSpec(shape=[None, 1, board_height * board_width],  dtype=tf.float32),
                                  tf.TensorSpec(shape=[None, 1, 1],  dtype=tf.float32)])
        def train(self, state_batch, prob_batch, score_batch):
            
            with tf.GradientTape() as tape:
                predictions = self.model(state_batch, training=True)  # Forward pass
                # the loss function is configured in `compile()`
                loss = self.model.compiled_loss([prob_batch, score_batch], predictions, regularization_losses=self.model.losses)
 
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))

            entropy = tf.negative(tf.reduce_mean(
               tf.reduce_sum(tf.exp(predictions[0]) * predictions[0], 2)))

            return (loss, entropy)



        @tf.function
        def export_param(self):
            args = []
            
            for weight in self.model.weights:
                args.append( tf.strings.join( [weight.name,
                    tf.io.encode_base64(tf.io.serialize_tensor(weight.read_value()))]
                    , ">"))
            
            encoded_str = tf.strings.join(args, "!")
            return encoded_str


        @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
        def import_param(self, encoded_str):
            args = tf.strings.split(encoded_str, sep=tf.convert_to_tensor('!'))
            for arg in args:
                pair = tf.strings.split( arg, sep=tf.convert_to_tensor('>'))
                tensor_value = tf.io.parse_tensor(tf.io.decode_base64(pair[1]), out_type=tf.float32)
                
                for weight in self.model.weights:
                    if tf.math.equal( tf.convert_to_tensor(weight.name), pair[0]):
                        weight.assign(tensor_value)
                        #tf.print( weight.name, "Assigned")
                
            return encoded_str
            
        

        @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
        def save(self, checkpoint_path):
            tensor_names = [weight.name for weight in self.model.trainable_variables]
            tensors_to_save = [weight.read_value() for weight in self.model.trainable_variables]
            tf.raw_ops.Save(
                filename=checkpoint_path, tensor_names=tensor_names,
                data=tensors_to_save, name='save')
            return checkpoint_path

        @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
        def restore(self, checkpoint_path):
            restored_tensors = {}
            for var in self.model.trainable_variables:
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


renju = create_model( 15, 15)

def save_model(folder_name):
    #Saving the model, explictly adding the concrete functions as signatures
    renju.model.save(folder_name, 
            save_format='tf', 
            overwrite=True,
            include_optimizer=True,
            signatures={
                'predict': renju.predict.get_concrete_function(), 
                'train' : renju.train.get_concrete_function(), 
                'save' : renju.save.get_concrete_function(),
                'restore' : renju.restore.get_concrete_function(),
                'random_choose_with_dirichlet_noice' : renju.random_choose_with_dirichlet_noice.get_concrete_function(),
                'export_param' : renju.export_param.get_concrete_function(),
                'import_param' : renju.import_param.get_concrete_function(),
            })

""""""
def train(state_batch, prob_batch, score_batch, lr):
    renju.lr.assign(lr)
    batch_size = tf.shape(state_batch)[0]
    state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
    prob_batch = tf.reshape(prob_batch, [batch_size, 225])
    score_batch = tf.reshape(score_batch, [batch_size, 1])

    loss = renju.model.evaluate(state_batch, [prob_batch, score_batch], batch_size=batch_size, verbose=0)
    action_probs, _ = renju.model.predict_on_batch(state_batch)

    entropy = -np.mean(np.sum(action_probs * np.log(action_probs + 1e-10), axis=1))
    
    renju.model.fit( state_batch, [prob_batch, score_batch], batch_size=batch_size)
    return (loss[0] + loss[1], entropy)

"""infer and return the first input result (probability and score)"""
def predict(state_batch):
    state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
    try:
        probs, scores = renju.model(state_batch)
    except:
        traceback.print_exc()
        raise
    # probs shape=(None, 225); scores shape=(None, 1), [[-0.14458223]]

     
    batch_size = tf.shape(state_batch)[0]
    if batch_size > 1:
        last = probs[0][0]
        index = 1
        all_same = True
        while index < batch_size:
            current = probs[index][0]
            index = index + 1
            if current != last:
                all_same = False
                break
            else:
                last = current
        if all_same:
            print( "WARNING: All predictions are the same ")

        last = scores[0]
        index = 1
        all_same = True
        while index < batch_size:
            current = scores[index]
            index = index + 1
            if current != last:
                all_same = False
                break
            else:
                last = current
        if all_same:
            print( "WARNING: All scores are the same.", scores)
        

    return probs[0].numpy().tolist(), scores[0][0].numpy()


def export_parameters():
    #print( pickle.loads(encoded) )
    return pickle.dumps(renju.model.get_weights(), 4)

def import_parameters(buffer):
    renju.model.set_weights(pickle.loads(buffer))
    return

def random_choose_with_dirichlet_noice(probs):
    return renju.random_choose_with_dirichlet_noice(tf.convert_to_tensor(probs))