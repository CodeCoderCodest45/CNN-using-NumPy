from builtins import range
import numpy as np

 

class Conv(object):
  @staticmethod
  def _find_roi(x, i, j, FH, FW, stride):
    """
    Extract the receptive field  or region of interest (ROI) from the single input tensor `x` for a given position (i, j) in the output feature map.
    The receptive field is the segment of the input tensor that the convolutional filter will overlap with during the convolution process. 
    Given a specific position (i, j) in the output feature map, the function determines and extracts the correspoding receptive field from the input tensor.
    Input:
        x (numpy.ndarray): Input data of shape (H, W, C).
        i (int): Vertical index of the output feature map.
        j (int): Horizontal index of the output feature map.
        FH (int): Height of the convolutional filter.
        FW (int): Width of the convolutional filter.
        stride (int): The number of pixels between adjacent receptive fields in the
                      horizontal and vertical directions.
    Returns:
       out: The receptive field with shape (FH, FW, C) from the input tensor `x` for a given position (i, j) in the output feature map.
    """

    hs = i * stride
    ws = j * stride
    out = x[hs:hs + FH, ws:ws + FW, :]

    return out  
    
  @staticmethod
  def naive_forward(x, w, b, conv_param):      
    """An implementation of the forward pass for a convolutional layer.

    The input consists of N data points, height H and
    width W, each with C channels. We convolve each input with F different filters, where each filter
    has height HH and width WW and spans all C channels.
    
    Input:
    - x: Input data of shape (N, H, W, C)
    - w: Filter weights of shape (F, FH, FW, C)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
        - 'stride': The number of pixels between adjacent receptive fields in the
            horizontal and vertical directions.
        - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros are placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. The original input x is not modified directly.

    Returns a tuple of:
    - out: Output data, of shape (N, H', W', F) where H' and W' are given by
      H' = 1 + (H + 2 * pad - FH) // stride
      W' = 1 + (W + 2 * pad - FW) // stride
    - cache: (x, w, b, conv_param)
    """

    N, H, W, C = x.shape
    F, FH, FW, _ = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']

    Hp = 1 + ((H + 2 * pad - FH) // stride)
    Wp = 1 + ((W + 2 * pad - FW) // stride)

    # Initialize the output tensor
    out = np.zeros((N, Hp, Wp, F))
    paddedx = np.pad(x, [(0, 0), (pad, pad), (pad, pad), (0, 0)], 'constant')

    for i in range(N):
        for j in range(F):
            for k in range(Hp):
                for m in range(Wp):
                    roi = Conv._find_roi(paddedx[i], k, m, FH, FW, stride)
                    out[i, k, m, j] = np.sum(roi * w[j]) + b[j]

    cache = (x, w, b, conv_param)

    return out, cache


  @staticmethod
  def forward(x, w, b, conv_param):      
    """An efficient implementation of the forward pass for a convolutional layer.

    The input consists of N data points, height H and
    width W, each with C channels. We convolve each input with F different filters, where each filter
    has height HH and width WW and spans all C channels.

    Input:
    - x: Input data of shape (N, H, W, C)
    - w: Filter weights of shape (F, FH, FW, C)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
         - 'stride': The number of pixels between adjacent receptive fields in the
           horizontal and vertical directions.
         - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, H', W', F) where H' and W' are given by
      H' = 1 + (H + 2 * pad - FH) // stride
      W' = 1 + (W + 2 * pad - FW) // stride
    - cache: (x, w, b, conv_param)
    """

    N, H, W, C = x.shape
    F, FH, FW, C = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']

    # Calculate the dimensions of the output feature map
    Hp = 1 + ((H + 2 * pad - FH) // stride)
    Wp = 1 + ((W + 2 * pad - FW) // stride)

    out = np.zeros((N, Hp, Wp, F))
    paddedx = np.pad(x, [(0, 0), (pad, pad), (pad, pad), (0, 0)], 'constant')

    w_flip = w.reshape(F, -1).T

    for i in range(N):
        for k in range(Hp):
            for m in range(Wp):
                roi = Conv._find_roi(paddedx[i], k, m, FH, FW, stride).reshape(1, -1)
                out[i, k, m, :] = np.dot(roi, w_flip) + b

    cache = (x, w, b, conv_param)

    return out, cache


  @staticmethod
  def backward(dout, cache):
    """An implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in Conv.forward

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """

    x, w, b, conv_param = cache
    N, H, W, C = x.shape
    F, FH, FW, C = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']

    Hp = 1 + ((H + 2 * pad - FH) // stride)
    Wp = 1 + ((W + 2 * pad - FW) // stride)

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.sum(dout, axis=(0, 1, 2))

    paddedx = np.pad(x, [(0, 0), (pad, pad), (pad, pad), (0, 0)], 'constant')
    padded_dx = np.pad(dx, [(0, 0), (pad, pad), (pad, pad), (0, 0)], 'constant')

    for i in range(N):
        for k in range(Hp):
            hs = k * stride
            for m in range(Wp):
                ws = m * stride
                window = paddedx[i, hs:hs + FH, ws:ws + FW, :]
                dw += np.sum(window * dout[i, k, m, np.newaxis, np.newaxis, np.newaxis, np.newaxis].T, axis=1)
                padded_dx[i, hs:hs + FH, ws:ws + FW, :] += np.sum(w * dout[i, k, m, np.newaxis, np.newaxis, np.newaxis].T, axis=0)

    dx = padded_dx[:, pad:pad + H, pad:pad + W, :]

    return dx, dw, db
 
class Pooling(object):
  @staticmethod
  def forward(x, pool_param):
    """An implementation of the forward pass for a max-pooling layer.
    Inputs:
    - x: Input data, of shape (N, H, W, C)
    - pool_param: dictionary with the following keys:
        - 'pool_size': The size of each pooling region.  
           Here, we only care about square sized pooling (pool_height == pool_weight).
        - 'stride': The distance between adjacent pooling regions
        - 'pool_type': "max" or "avg"

    No padding is necessary here
    Returns a tuple of:
    - out: Output data, of shape (N,  H', W', C) where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """

    N, H, W, C = x.shape
    pool_size = pool_param['pool_size']

    stride = pool_param['stride']

    # dimension of filters
    Hp = (H - pool_size) // stride + 1
    Wp = (W - pool_size) // stride + 1

    # empty pooling layer
    out = np.zeros((N, Hp, Wp, C))
    Hc = H % pool_size
    Wc = W % pool_size

    # if integer number of pooling patches cover the images the following fast forward method
    # is used, if not naive implementaion is used (same for backward pass)
    # in addition, since average pooling is not used in 'numpy_cnn' model, computational
    # efficieny is not considered


    # fast
    if (not Hc) and (not Wc) and pool_param['pool_type'] == 'max':
        xp = x.reshape(N, H // pool_size, pool_size, W // pool_size, pool_size, C)
        out = xp.max(axis=2).max(axis=3)

    # naive
    else:
        for j in range(Hp):
            for k in range(Wp):

                if pool_param['pool_type'] == 'max':
                    out[:, j, k, :] = (
                        x[:, j * stride:(j * stride + pool_size), k * stride:(k * stride + pool_size), :].max(axis=(1, 2)))

                elif pool_param['pool_type'] == 'avg':
                    out[:, j, k, :] = np.mean(
                        x[:, j * stride:(j * stride + pool_size), k * stride:(k * stride + pool_size), :], axis=(1, 2))

    cache = (x, pool_param)

    return out, cache

  @staticmethod
  def backward(dout, cache):
    """An implementation of the backward pass for a pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """

    x, pool_param = cache
    N, H, W, C = x.shape
    pool_size = pool_param['pool_size']
    stride = pool_param['stride']

    # dimensions of output
    Hp = 1 + (H - pool_size) // stride
    Wp = 1 + (W - pool_size) // stride

    # initial gradient dx
    dx = np.zeros_like(x)

    Hc = H % pool_size #in this case H and W are equal (32), but this helps with more general cases
    Wc = W % pool_size

    if (not Hc) and (not Wc) and pool_param['pool_type'] == 'max':
        # fast back pass
        # storing out and x_reshaped during forward pass would have been efficient but
        # the skeleton code is already provided

        xp = x.reshape(N, H // pool_size, pool_size, W // pool_size, pool_size, C)
        out = xp.max(axis=2).max(axis=3)
        dxp = np.zeros_like(xp)
        outp = out[:, :, np.newaxis, :, np.newaxis, :]
        mask = (xp == outp)
        doutp = dout[:, :, np.newaxis, :, np.newaxis, :]
        dout_stack, _ = np.broadcast_arrays(doutp, dxp)
        dxp[mask] = dout_stack[mask]
        dxp /= np.sum(mask, axis=(2, 4), keepdims=True)
        dx = dxp.reshape(x.shape)

    else:

        # naive
        for i in range(Hp):
            for j in range(Wp):
                if pool_param['pool_type'] == 'max':
                    patch = x[:, i * stride:i * stride + pool_size, j * stride:j * stride + pool_size, :]
                    max_pos = np.max(patch, axis=(1, 2), keepdims=True) == patch
                    dx[:, i * stride:i * stride + pool_size, j * stride:j * stride + pool_size, :] += max_pos * (dout[:, i, j, :])[:, np.newaxis, np.newaxis, :]
                elif pool_param['pool_type'] == 'avg':
                    hs, he = i * stride, i * stride + pool_size
                    ws, we = j * stride, j * stride + pool_size
                    dx[:, hs:he, ws:we, :] += (dout[:, i, j, np.newaxis, np.newaxis, :] / (pool_size * pool_size))

    return dx

class BatchNorm(object):
    @staticmethod
    def _compute_means_and_vars(x, axis):
        """
        Computes the mean and variance of the input data.
        
        Inputs:
        - x: Input data.
        - axis: Axis or axes along which the means and variances are computed.

        Returns:
        - means: Computed means along specified axis.
        - vars: Computed variances along specified axis.
        """

        means = np.mean(x, axis=axis)
        vars = np.var(x, axis=axis)

        return means, vars

    @staticmethod
    def _normalize_data(x, means, vars, eps):
        """
        Normalizes the input data using the provided means and variances.
        
        Inputs:
        - x: Input data.
        - means: Means used for normalization.
        - vars: Variances used for normalization.
        - eps: Small value to add to variance for numerical stability.

        Returns:
        - out: Normalized input 'x'
        """

        out = (x - means) / np.sqrt(vars + eps)

        return out

    @staticmethod
    def _scale_and_shift(x, gamma, beta):
        """
        Scales and shifts the normalized data using gamma and beta parameters.
        Inputs:
        - x: Data that has been normalized.
        - gamma: Scaling parameter.
        - beta: Shifting parameter.

        Returns:
        - out: Scaled and shifted data.
        """

        out = gamma * x + beta

        return out

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Computes the forward pass of batch normalization for CNN.
        Implement it using above functions (_compute_means_and_vars, _normalize_data, _scale_and_shift)

        Inputs:
        - x: Input data of shape (N, H, W, C)
        - gamma: Scale parameter, of shape (C,)
        - beta: Shift parameter, of shape (C,)
        - bn_param: Dictionary with the following keys:
            - mode: 'train' or 'test'; required
            - eps: Constant for numeric stability (we set for 1e-5 here)
            - momentum: Constant for running mean / variance. momentum=0 means that
                old information is discarded completely at every time step, while
                momentum=1 means that new information is never incorporated. The
                default of momentum=0.9 should work well in most situations.
            - running_mean: Array of shape (C,) giving running mean of features
            - running_var Array of shape (C,) giving running variance of features

        Returns a tuple of:
        - out: Output data, of shape (N, H, W, C)
        - cache: Values needed for the backward pass
        """
        out, cache = None, None
        
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)
        N, H, W, C = x.shape
        running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x.dtype))
        running_var = bn_param.get('running_var', np.zeros(C, dtype=x.dtype))


        if mode == 'train':
            #################################WRITE YOUR CODE###########################################
            # TODO: Implement the training-time forward pass for spatial batch norm.      
            # - During training, compute the mean and variance from the input minibatch.
            # - Normalize the data using these statistics.
            # - Scale and shift the normalized data with gamma and beta.
            # - Calculate the running mean/ std and store them         
            #
            # - Store the output in the variable out, and intermediates              
            #   that you need for the backward pass should be stored in the cache variable.                                                           
            # For further information, refer to the original paper (https://arxiv.org/abs/1502.03167)  
            #  You should use above functions (_compute_means_and_vars, _normalize_data, _scale_and_shift)
            ##########################################################################################


            means, vars = BatchNorm._compute_means_and_vars(x, axis=(0, 1, 2))
            normx = BatchNorm._normalize_data(x, means, vars, eps)
            xhat = BatchNorm._scale_and_shift(normx, gamma, beta)

            running_mean = (1 - momentum) * running_mean + momentum * means
            running_var = (1 - momentum) * running_var + momentum * vars
            out = xhat
            cache = (x, gamma, beta, means, vars, eps, normx)

            ###########################################################################################
            #                                  END OF YOUR CODE                            
            ###########################################################################################
        elif mode == 'test':
            #################################WRITE YOUR CODE###########################################
            # TODO: Implement the test-time forward pass for batch norm. 
            # - Use the running mean and variance to normalize the incoming data,   
            #   then scale and shift the normalized data using gamma and beta.      
            # - Store the result in the out variable.                               
            ###########################################################################################

            normx = (x - running_mean) / np.sqrt(running_var + eps)
            out = BatchNorm._scale_and_shift(normx, gamma, beta)

            ###########################################################################################
            #                                  END OF YOUR CODE                           
            ###########################################################################################
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var        
        
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass of batch normalization for CNN.

        Inputs:
        - dout: Upstream derivatives, of shape (N, H, W, C)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient with respect to inputs, of shape (N, H, W, C)
        - dgamma: Gradient with respect to scale parameter, of shape (C,)
        - dbeta: Gradient with respect to shift parameter, of shape (C,)
        """

        x, gamma, beta, means, vars, eps, normx = cache
        N, H, W, C = x.shape
        
        # gradients for beta and gamma
        dbeta = np.sum(dout, axis=(0, 1, 2))
        dgamma = np.sum(dout * normx, axis=(0, 1, 2))

        # gradient for dx
        d_normx = dout * gamma
        dvars = np.sum(d_normx * (x - means) * (-0.5) * np.power(vars + eps, -1.5), axis=(0, 1, 2))
        dmeans = np.sum(d_normx * (-1 / np.sqrt(vars + eps)), axis=(0, 1, 2))
        dx = d_normx * 1 / np.sqrt(vars + eps) + dvars * 2 * (x - means) / (N * H * W) + dmeans / (N * H * W)

        return dx, dgamma, dbeta

