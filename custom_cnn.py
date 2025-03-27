import torch.nn as nn
from helper_functions import *
from PIL import Image

class numpy_CNN(object):
    """
    A three-layer convolutional network with the following architecture:

    First layer: conv - batch norm- relu - 2x2 max pool
    Second layer:  affine - relu 
    Last layer: affine - softmax

    The network operates on minibatches of data that have shape (N, H, W, C)
    - N: number of images
    - H: image height
    - W: image width
    - C: number of input channels
    
    """

    def __init__(
        self,
        input_dim=(32, 32, 3),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (H, W, C) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        H, W, C = input_dim
        F, FH, FW = num_filters, filter_size, filter_size

        # conv layer
        self.params['W1'] = weight_scale * np.random.randn(F, FH, FW, C)
        self.params['b1'] = np.zeros(F)
        self.params['gamma1'] = np.ones(F)
        self.params['beta1'] = np.zeros(F)

        # affine layer
        self.params['W2'] = weight_scale * np.random.randn(F * H * W // 4, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)

        # output layer
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: 
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # Extract Batch Norm parameters
        gamma1, beta1 = self.params['gamma1'], self.params['beta1']
        bn_param={"mode": "train"}

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = { "pool_size": 2, "stride": 2, "pool_type": "max"}

        if y is None:
            bn_param={"mode": "test"} 
            
        out, conv_cache = conv_bn_relu_pool_forward(X, W1, b1, gamma1, beta1, conv_param, pool_param, bn_param)  #CPOINT
        out, hidden_cache = affine_relu_forward(out, W2, b2)

        # final layer
        scores, out_cache = affine_forward(out, W3, b3)

        if y is None:
            return scores

        loss, grads = 0, {}
        loss, ds = softmax_loss(scores, y)

        # add reg loss
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))

        # backprop from the last to first layer
        dout, grads['W3'], grads['b3'] = affine_backward(ds, out_cache)
        grads['W3'] += self.reg * W3

        # hidden layer
        dout, grads['W2'], grads['b2'] = affine_relu_backward(dout, hidden_cache)
        grads['W2'] += self.reg * W2

        # conv (lowest layer)
        _, grads['W1'], grads['b1'], grads['gamma1'], grads['beta1'] = conv_bn_relu_pool_backward(dout, conv_cache)   
        grads['W1'] += self.reg * W1

        return loss, grads
    
    
class pytorch_CNN(nn.Module):
    def __init__(self, num_class=10):
        super(pytorch_CNN, self).__init__()
        """
        A convolutional neural network (using pytorch) with the following architecture:
        - Feature Extraction Layer Block
          - first layer: Conv, Batchnorm, Relu, Max pool
          - second layer:  Conv, Batchnorm, Relu, Max pool
        - Classification Layer Block
          - first layer: Linear, Relu
          - second layer: Linear, Relu
          - final layer: Linear
        """

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(4, 4), padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features= 16*8*8, out_features=4096),
            nn.ReLU(inplace=True)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Linear(in_features=4096, out_features=num_class)
          
    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.out(x)
        
        return output

