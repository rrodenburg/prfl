import numpy as np
import tensorflow as tf
import random

memory = np.load('../../data/repl_mem.npy')

print(np.take(memory, indices = 0, axis = 1)[0].shape)

### Define neural network ### https://www.tensorflow.org/tutorials/layers

learning_rate = 0.001
frame_stacks = 4
width = 84
heigth = 84

#def convnet(mode):
#    x = tf.placeholder("int32",(-1,width,heigth,frame_stacks)) #input states
#    y = tf.placeholder("float32",(-1)) #y-values for loss function, as described in atari paper
#    a = tf.placeholder("int32",(-1)) #actions played in batch; 0: nothing, 1: up, 2: down
#
#    # Convolutional Layer #1
#    conv1 = tf.layers.conv2d(
#      inputs = x,
#      filters = 5,
#      kernel_size = [8, 8],
#      strides = (4,4),
#      padding = "valid", #valid means no padding
#      activation = tf.nn.relu)
#
#    # Output layer (dense layer)
#    conv1_flat = tf.reshape(conv1,[-1,20*20*5])
#    output = tf.layers.dense(inputs=conv1_flat, units = 3)
#    best_action = tf.argmax(input=output, axis=1)

#    predictions = {
#      # Generate predictions (for PREDICT and EVAL mode)
#        "best_action": tf.argmax(input=output, axis=1), 
#        "max_Q_value": tf.reduce_max(input=output,axis=1, name = 'Q_max')
#      }
#
#    if mode == tf.estimator.ModeKeys.PREDICT:
#        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
#
#    # loss function
#    onehot_actions = tf.one_hot(indices=a, depth=3)#batch-actions converted, s.t. 0 = nothing --> (1,0,0) etc
#    onehot_actions = tf.transpose(onehot_actions)
#    Q_values = tf.diag_part(tf.matmul(output,onehot_actions))
#    loss = tf.reduce_mean(tf.square(y - Q_values),axis=0)
#
#  # Configure the Training Op (for TRAIN mode)
#    if mode == tf.estimator.ModeKeys.TRAIN:
#        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,momentum = 0.95,epsilon=0.01)
#        train_op = optimizer.minimize(
#            loss=loss,
#            global_step=tf.train.get_global_step())
#        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


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

x = tf.placeholder("float32",(None,width,heigth,frame_stacks)) #input states
y = tf.placeholder("float32",(None)) #y-values for loss function, as described in atari paper
a = tf.placeholder("float32",(None)) #actions played in batch; 0: nothing, 1: up, 2: down
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
best_action = tf.argmax(input=output, axis=1)
max_Q_value = tf.reduce_max(output,axis=1, name = 'Q_max')

#loss function
onehot_actions = tf.one_hot(indices=a, depth=3)#batch-actions converted, s.t. 0 = nothing --> (1,0,0) etc
onehot_actions = tf.transpose(onehot_actions)
Q_values = tf.diag_part(tf.matmul(output,onehot_actions))
loss = tf.reduce_mean(tf.square(y - Q_values),axis=0)

optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,momentum = 0.95,epsilon=0.01)

global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(
            loss=loss,
            global_step=global_step)

model_params = {
	'learning_rate' : learning_rate
				}

#Q_estimator = tf.estimator.Estimator(
#    	model_fn=convnet, params=model_params)#, model_dir="/tmp/convnet_model")

predict_x = np.take(memory, indices = 3, axis = 1)[:32]
predict_x = [x.astype('float32') for x in predict_x]
#print(predict_x.shape)

print(type(memory))
idx = np.random.randint(len(memory), size = 32)
train = memory[idx]
print(train[0])
#train_x = np.array([x.astype('float32') for x in np.take(train, indices = 0, axis = 1)])
train_x = np.take(train, indices = 0, axis = 1)
train_x = [x.astype('float32') for x in train_x]
train_y = [x[3].astype('float32') for x in train]

print(len(predict_x), len(predict_x[0]), predict_x[0].shape, predict_x[0])
print(len(train_x), len(train_x[0]), train_x[0].shape, type(train_x[0]))

trans_dict = {
			  'stay' : 0,
			  'up' : 1,
			  'down': 0
			  }

train_action = [trans_dict[x[1]] for x in train]
print(train_action)
#print(train_y)

#predict_input_fn = tf.estimator.inputs.numpy_input_fn(
#    x={"x": predict_x},
#    num_epochs=1,
#    shuffle=False)


#predictions = list(Q_estimator.predict(input_fn=predict_input_fn))
#print(predictions)



sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
best_action, q_value = sess.run([best_action, max_Q_value], {x: predict_x})
print(best_action, q_value)

feed_dict_train = {
					x : train_x,
					y : q_value,
					a : train_action
					}

sess.run(train_op)
loss_value = sess.run([loss], feed_dict = feed_dict_train)
print(loss_value)