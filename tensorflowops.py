'''
By Adithya Ramachandran
Step by Step Procedure of how the neural network is built :

- Get the input data ( Using Mnist for this example)
- The data is then weighed and then sent to the first hidden layer
- From the hidden layer, the weighed data is then sent to an activation function
- The data is then sent to hidden layer 2 along with some new weights. 
- The hidden layer 2 will have its own weights and the data from hidden layer 2 will then be sent to 
  output layer along with weights. 
- The output from the output layer is compared with the intended output using a loss function ( in this case, it is cross-entropy)
- An optimization function is used to minimize the cost. (Ex: AdamOptimizer)
- Using the optimization function we would be able to alter the weights and correct the output ( to get the intended output)

  
'''
import tensorflow as tf
from tensorflow.examples.tutorials.Mnist import input_data

mnist = input_data.read_data_sets("/tmp/data",one_hot=True)

# Defining nodes for both the hidden layers.  
nodes_hiddenLayer_1 = 500
nodes_hiddenLayer_2 = 500

classes = 10
batch_size = 100

# Placeholder for x and y. 
# x will have a fixed column size of 784 (i.e 28*28) and will take any number of rows
x= tf.placeholder('float',[None,784])
y = tf. placeholder('float')

def model_NN(data):
	'''
	Model : (input data * weights) + biases
	activation function : relu

	'''

	hiddenLayer_1 = {'weights':tf.Variable(tf.random_normal([784,nodes_hiddenLayer_1])), 'biases':tf.Variable(tf.random_normal([nodes_hiddenLayer_1]))}
	hiddenLayer_2 = {'weights':tf.Variable(tf.random_normal([nodes_hiddenLayer_1,nodes_hiddenLayer_2])), 'biases':tf.Variable(tf.random_normal([nodes_hiddenLayer_2]))}
	outputLayer = {'weights':tf.Variable(tf.random_normal([nodes_hiddenLayer_2,classes])), 'biases':tf.Variable(tf.random_normal([classes]))}

	layer_1 = tf.add(tf.matmul(data, hiddenLayer_1['weights']), hiddenLayer_1['biases'])
	layer_1= tf.nn.relu(layer_1)

	layer_2 = tf.add(tf.matmul(layer_1, hiddenLayer_2['weights']), hiddenLayer_2['biases'])
	layer_1= tf.nn.relu(layer_2)

	output = tf.add(tf.matmul(layer_2, outputLayer['weights']), outputLayer['biases'])

	return output

def train_NN(x):

	prediction = model_NN(data)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	number_of_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epochs in range(number_of_epochs):
			epoch_loss = 0 
			for _ in range(int(mnist.train.num_examples/batch_size)):
				x,y = mnist.train.next_batch(batch_size)
				_,c = sess.run([optimizer,cost], feed_dict = {x: x,y : y})
				epoch_loss += c
			print(epoch,'Epoch completed out of', number_of_epochs, 'loss:',epoch_loss)

		correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))

		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('Accuracy:',accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))


train_NN(x)















