import torch
import random

# generate the dataset
def synthetic_data(weight, bias, num_examples):
  """ weight is vector of size (n,) and bias is a number
  """
  X = torch.normal(0,1,(num_examples,len(weight))) # generate X with shape ()
  Y = torch.matmul(X,weight) + bias
  Y += torch.normal(0,0.001,(num_examples,))
  return X,Y.reshape((-1,1))

# data loader
def data_loader(batch_size, features, labels):
  """

  :param batch_size:  the size of each data batch
  :param features: the features aka the X to feed to the model
  :param labels: the label aka the y, target of the dataset
  :return: features and label divided into batch after shuffle
  """
  num_examples = features.shape[0]
  indicies = list(range(num_examples))
  random.shuffle(indicies)
  for i in range(0,num_examples,batch_size):
    batch_indicies = torch.tensor(indicies[i : min(i + batch_size, num_examples)])
    yield features[batch_indicies],labels[batch_indicies]

