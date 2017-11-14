import numpy as np
import tensorflow as tf

memory = np.load('repl_mem.npy')

print(np.take(memory, indices = 0, axis = 1)[0].shape)

### Define neural network ### https://www.tensorflow.org/tutorials/layers

def convnet(mode):
    x = tf.placeholder("int32",(-1,width,heigth,frame_stacks)) #input states
    y = tf.placeholder("float32",(-1)) #y-values for loss function, as described in atari paper
    a = tf.placeholder("int32",(-1)) #actions played in batch; 0: nothing, 1: up, 2: down

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs = x,
      filters = 5,
      kernel_size = [8, 8],
      strides = (4,4),
      padding = "valid", #valid means no padding
      activation = tf.nn.relu)

    # Output layer (dense layer)
    conv1_flat = tf.reshape(conv1,[-1,20*20*5])
    output = tf.layers.dense(inputs=conv1_flat, units = 3)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
        "best_action": tf.argmax(input=output, axis=1), 
        "max_Q_value": tf.reduce_max(input=output,axis=1, name = 'Q_max')
      }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # loss function
    onehot_actions = tf.one_hot(indices=a, depth=3)#batch-actions converted, s.t. 0 = nothing --> (1,0,0) etc
    onehot_actions = tf.transpose(onehot_actions)
    Q_values = tf.diag_part(tf.matmul(output,onehot_actions))
    loss = tf.reduce_mean(tf.square(y - Q_values),axis=0)

  # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,momentum = 0.95,epsilon=0.01)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


#def main():
#	train_data = np.take(memory, indices = 0, axis = 1)
#
#	# Create the Estimator
#	mnist_classifier = tf.estimator.Estimator(
#    	model_fn=convnet, model_dir="/tmp/convnet_model")
#
#	# Set up logging for predictions
#  	tensors_to_log = {"max_Q_value": "Q_max"}
#  	logging_hook = tf.train.LoggingTensorHook(
#      tensors=tensors_to_log, every_n_iter=50)


# Generate predictions https://www.tensorflow.org/get_started/estimator
'''
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": new_samples},
    num_epochs=1,
    shuffle=False)

predictions = list(classifier.predict(input_fn=predict_input_fn))
predicted_classes = [p["classes"] for p in predictions]
'''

Q_estimator = tf.estimator.Estimator(
    	model_fn=convnet, model_dir="/tmp/convnet_model")

predict_x = np.take(memory, indices = 3, axis = 1)[:32]

predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": predict_x},
    num_epochs=1,
    shuffle=False)


predictions = list(Q_estimator.predict(input_fn=predict_input_fn))
print(predictions)