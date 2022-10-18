# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in Tensorflow V2
Author : Jerry.Wang  vcer#qq.com
"""
import traceback
import pickle
import pathlib
import numpy as np
import platform
import tensorflow as tf
import tensorflow_probability as tfp

print( "platform: ", platform.platform() )
print( "tensorflow version:", tf.__version__ )
print( "tensorflow_probability version:", tfp.__version__ )


def create_model(board_width, board_height):

    l2_penalty_beta = 1e-4
    data_format="channels_last"

    class ResBlock(tf.keras.Model):
        def __init__(self, filters):
            super().__init__()

            self.shortcut = tf.keras.Sequential()

            self.conv1 = tf.keras.layers.Conv2D(filters, 
                kernel_size=(3, 3), 
                strides=1, 
                data_format=data_format, 
                kernel_regularizer=tf.keras.regularizers.L2(l2_penalty_beta),
                padding='same')

            self.bn1 = tf.keras.layers.BatchNormalization()

            self.relu1 = tf.keras.layers.LeakyReLU(alpha=0.1)

            self.conv2 = tf.keras.layers.Conv2D(filters, 
                kernel_size=(3, 3), 
                strides=1, 
                data_format=data_format, 
                kernel_regularizer=tf.keras.regularizers.L2(l2_penalty_beta),
                padding='same')
            self.bn2 = tf.keras.layers.BatchNormalization()
            
            self.relu = tf.keras.layers.LeakyReLU(alpha=0.1)
        def call(self, input):
            shortcut = self.shortcut(input)

            input = self.conv1(input)
            input = self.bn1(input)
            input = self.relu1(input)

            input = self.conv2(input)
            input = self.bn2(input)

            input = input + shortcut
            return self.relu(input)

    class RenjuModel(tf.Module):
        def __init__(self):
            

            # Define the tensorflow neural network
            # 1. Input:
            self.inputs = tf.keras.Input( shape=(4, board_height, board_width), dtype=tf.dtypes.float32, name="input")

            

            # convert from  NCHW(channels_first) to NHWC(channels_last) because channels_first is not supported by CPU
            if data_format == 'channels_last':
                self.source = tf.keras.layers.Permute((2,3,1))(self.inputs)
            else:
                self.source = self.inputs

             # 2. Common Networks Layers
            self.conv1 = tf.keras.layers.Conv2D( name="conv1",
                filters=32,
                kernel_size=(3, 3),
                padding="same",
                data_format=data_format,
                kernel_regularizer=tf.keras.regularizers.L2(l2_penalty_beta)
                )(self.source)

            self.reslayer = tf.keras.Sequential([
                ResBlock(32),
                ResBlock(32),
                ResBlock(32),
                ResBlock(32),
                ResBlock(32),
                ResBlock(32),
                ResBlock(32),
                ResBlock(32),
                ResBlock(32),
                ResBlock(32),
                ResBlock(32),
                ResBlock(32),
                ResBlock(32),
                ResBlock(32),
                ResBlock(32),
                ResBlock(32),
                ResBlock(32),
                ResBlock(32),
                ResBlock(32),
            ], name='resblocks')(self.conv1)

            # 3-1 Action Networks
            self.action_conv = tf.keras.layers.Conv2D( name="action_conv",
                filters=2,
                kernel_size=(1, 1),
                data_format=data_format,
                kernel_regularizer=tf.keras.regularizers.L2(l2_penalty_beta)
                )(self.reslayer)

            self.action_bn = tf.keras.layers.BatchNormalization(name="action_bn")(self.action_conv)

            self.action_act = tf.keras.layers.LeakyReLU(0.1, name="action_activation")(self.action_bn)

            # flatten tensor
            self.action_conv_flat = tf.keras.layers.Flatten(name="action_flatten")(self.action_act)

            # 3-2 Full connected layer, the output is the log probability of moves
            # on each slot on the board
            self.action_fc = tf.keras.layers.Dense( board_height * board_width,
                activation=tf.keras.activations.softmax,
                name="action_fc",
                kernel_regularizer=tf.keras.regularizers.L2(l2_penalty_beta)
                )(self.action_conv_flat)


            # 4 Evaluation Networks
            self.evaluation_conv = tf.keras.layers.Conv2D( name="evaluation_conv",
                filters=1,
                kernel_size=(1, 1),
                data_format=data_format,
                kernel_regularizer=tf.keras.regularizers.L2(l2_penalty_beta)
                )(self.reslayer)

            self.evaluation_bn = tf.keras.layers.BatchNormalization(name="evaluation_bn")(self.evaluation_conv)

            self.evaluation_act = tf.keras.layers.LeakyReLU(0.1, name="evaluation_activation")(self.evaluation_bn)

            self.evaluation_conv_flat = tf.keras.layers.Flatten(name="evaluation_flatten")(self.evaluation_act)

            self.evaluation_fc1 = tf.keras.layers.Dense( 256,
                name="evaluation_fc1",
                activation=tf.keras.activations.elu,
                kernel_regularizer=tf.keras.regularizers.L2(l2_penalty_beta)
                )(self.evaluation_conv_flat)

            self.evaluation_fc2 = tf.keras.layers.Dense( 1, 
                activation=tf.keras.activations.tanh,
                name="evaluation_fc2",
                kernel_regularizer=tf.keras.regularizers.L2(l2_penalty_beta)
                )(self.evaluation_fc1)

            self.model = tf.keras.Model(inputs=self.inputs, outputs=[self.action_fc, self.evaluation_fc2], name="renju_model")             

            self.model.summary()
 
            self.lr = tf.Variable(0.001, trainable=False, dtype=tf.dtypes.float32)

            # loss = (z - v)^2 + pi^T * log(p) + c||theta||^2
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = self.lr),
                    loss=[tf.keras.losses.CategoricalCrossentropy(), tf.keras.losses.MeanSquaredError()])



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

    
        @tf.function(input_signature=[
            tf.TensorSpec([None, 4, board_height, board_width], tf.float32),
        ])
        def predict(self, state_batch):
            probs, scores = renju.model(state_batch)
            # probs shape=(None, 225); scores shape=(None, 1), [[-0.14458223]]
            return probs[0], scores[0][0]

        @tf.function(input_signature=[
            tf.TensorSpec([None, 4, board_height, board_width], tf.float32),
        ])
        def predict_batch(self, state_batch):
            probs, scores = renju.model(state_batch)
            # probs shape=(None, 225); scores shape=(None, 1), [[-0.14458223]]
            return probs, scores
        


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
    action_probs, pred_scores = renju.model.predict_on_batch(state_batch)

    last = pred_scores[0]
    index = 1
    all_same = True
    while index < batch_size:
        current = pred_scores[index]
        index = index + 1
        if current != last:
            all_same = False
            break
        else:
            last = current
    if all_same:
        print( "WARNING: All scores are the same.")

    entropy = -np.mean(np.sum(action_probs * np.log(action_probs + 1e-10), axis=1))
    
    renju.model.fit( state_batch, [prob_batch, score_batch], batch_size=batch_size, verbose=0)
    return (loss[0], entropy)

def predict_batch(state_batch):
    state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
    try:
        probs, scores = renju.model.predict_on_batch(state_batch)
    except:
        traceback.print_exc()
        raise
    # probs shape=(None, 225); scores shape=(None, 1), [[-0.14458223]]

    return probs.tolist(), scores.tolist()

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
       

    return probs[0].numpy().tolist(), scores[0][0].numpy()


def export_parameters():
    #print( pickle.loads(encoded) )
    return pickle.dumps(renju.model.get_weights(), 4)

def import_parameters(buffer):
    renju.model.set_weights(pickle.loads(buffer))
    return

def save_model(folder_name):
    #Saving the model, explictly adding the concrete functions as signatures
    renju.model.save(folder_name, 
            save_format='tf', 
            overwrite=True,
            include_optimizer=True,
            signatures={
                'predict': renju.predict.get_concrete_function()
            })


def save_quantized_model(file_name):
    with open("sample_input.pickle", "rb") as input_file:
        fileContent = input_file.read()
        state_tensor_batch = to_list(pickle.loads(fileContent)['state_tensor_batch'])

    # You need to provide either a dictionary with input names and values, a tuple with signature key and a dictionary with input names and values, 
    # or an array with input values in the order of input tensors of the graph in the representative_dataset function. 
    def representative_data_gen():
        for input_value in state_tensor_batch:
            yield [tf.convert_to_tensor([input_value])]

    converter = tf.lite.TFLiteConverter.from_keras_model(renju.model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    # https://www.tensorflow.org/lite/performance/post_training_integer_quant
    # converter.representative_dataset = representative_data_gen

    # https://www.tensorflow.org/lite/performance/post_training_integer_quant_16x8
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    
    tflite_model = converter.convert()

    tflite_models_dir = pathlib.Path("./")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    # Save the quantized model:
    tflite_model_quant_file = tflite_models_dir/file_name
    tflite_model_quant_file.write_bytes(tflite_model)

def to_list(x):
    ls = list(x)
    for idx, sub in enumerate(ls):
        if type(sub) == tuple:
            ls[idx] = to_list(sub)
    return ls

#with open("latest.weights", mode='rb') as file:
#   buffer = file.read()
#   import_parameters(buffer)
#save_quantized_model('best.tflite')


"""
from pprint import pprint
from keras.models import Model

with open("/Users/jerry/projects/renju/renju.git/game/target/debug/data/1.pickle", mode='rb') as file:
    buffer = file.read()
    input = pickle.loads(buffer)
    ls = to_list(input['state_tensor_batch'])
    input = [ ls[1], ls[2] ]
    predict_batch(input)[1]

    print(renju.model.input)

    
    for layer in renju.model.layers:
        print(layer.name)
        if layer.name == 'evaluation_flatten':
            YY = layer.output

    XX = renju.model.input
    proxy_model = Model(XX, YY)

    pprint( proxy_model.predict(input) )
"""
