#Load the MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#MNIST data split into 3 parts, 55,000 points of training data, 10,000 points of test data
#and 5,000 points of validation data

#flatten the 28*28 grid of an image into a 1-d vector which is 784 by 1
#Softmax regression doesn't take advantage of the image's 2-d structure

#So the training data is a tensor(n-dimensional array) with the shape [55,000, 784]
#basically 55,000 images and each image is a 784 by 1 vector(since we turned the 2d picture into 1d)

#we are converting our vectors to one-hot vectors.
#A one hot vector has 0 in most dimensions and 1 in a single dimension

#The training labels are a vector [55,000, 10]. Each of the 55,000 images could be one of 10 numbers

#Softmax regression
    #Great for assigning probabilities to an oject being one of several different things

#Softmax has 2 steps, (1) adding up the evidence for our input being a certain class, and then converting the evidence into probabilities

#In order to tally up the evidence that a given image is in a class, we do a
#weighted sum of the pixel intensities. The weight is negatie if that pixel having a
#high intensity is evidence against that image being in the class.
#Weight positive if the pixel is in favor of being in favor of being in that class

#We add in extra evidence called Bias, Bias is stuff that is independent of input

#Softmax first exponentiates it input(applies weights and bais to hypothesis) and then
#normalizes the output to give probabilities between 0 and 1


#importing tensorflow
import tensorflow as tf

#creating a variable to serve as a placeholder for tensorflow to put the flattened MNIST
#images into a 784 dimensional vector
x = tf.placeholder(tf.float32, [None, 784])

#The W is the weights for our model, and the b is the bias for the model
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
#initializing W and b as 0 since our model will learn what W and b are

#implementing our model
y = tf.nn.softmax(tf.matmul(x, W) + b)

#In order to train our model we need to define what it means to be good/bad
    #we will use a cost function, or loss function

#A commong function to determine the loss of a model is called "cross-entropy"
#cross entropy measures how inefficient our predictions are for describing the truth

#to implement cross entropy we need add placeholders to input the correct answers
y_ = tf.placeholder(tf.float32, [None, 10])

#implementing the cross entropy function below(numerically unstable in tensor flow so
#Below code unstable, use tf.nn.softmax_cross_entropy_with_logits instead#
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#tf.nn.softmax_cross_entropy_with_logits

#Since tensorflow knows the entire graph of computations, it can automatically, use
#the backpropagation algorithm to efficiently determine how variables affect loss
#it can then apply your choice optimization algorithm(in this case gradient descent)
#to modify the variables and reduce loss
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
#we are using gradient descent, a simple procedure where tensorflow shifts each variable
#a little bit in the direction that reduces the cost.

#launch the interactive session
sess = tf.InteractiveSession()

#initialize the variables that we created
tf.global_variables_initializer().run()

#training with 1,000 times!
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#for each loop of the for-loop we get a "batch" of 100 random datapoints from our training set.
#we run train_step feeding in the batches to replace the placeholders.

#Using small batches of random data is called stochastic training
    #in our case stochastic gradient descent

#tf.argmax is super useful because it gives you the index of the highest entry in a tensor along an axis
#so tf.argmax(y, 1) is the label our model thinks is most likely for each input
#tf.argmax(y_, 1) is the actual/correct label

#we can use tf.equal to check if our prediction matches the truth
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#gives us a list of booleans, to determine what fraction are correct we cast to floating point numbers
#after casting to floats, we take the mean, for example
#if we were returned [True, False, True, True]
#we would cast to [1, 0, 1, 1], and after casting to float this would be a 0.75 accuracy

#asking for accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#printing results
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#should get accuracy of about 92%

  




  
  











