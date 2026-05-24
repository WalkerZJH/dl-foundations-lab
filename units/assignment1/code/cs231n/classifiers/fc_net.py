from builtins import range
from builtins import object
import os
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    
    中文版本：一个两层的全连接神经网络，使用ReLU非线性和softmax损失，采用模块化层设计。假设输入维度为D，隐藏维度为H，并进行C类分类。
    结构应为仿射-RELU-仿射-Softmax。
    注意：此类不实现梯度下降；相反，它将与一个单独的Solver对象交互，该对象负责运行优化。
    模型的可学习参数存储在字典self.params中，该字典将参数名称映射到numpy数组。
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.、
        中文版本：
        初始化一个新的网络。
        输入：
        - input_dim：一个整数，表示输入的大小
        - hidden_dim：一个整数，表示隐藏层的大小
        - num_classes：一个整数，表示要分类的类别数
        - weight_scale：标量，给出权重随机初始化的标准差。
        - reg：标量，给出L2正则化强度。
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        # 中文版本要求：
        # 初始化两层网络的权重和偏置。权重应从以0.0为中心的高斯分布初始化，标准差等于weight_scale，
        # 偏置应初始化为零。所有权重和偏置应存储在字典self.params中，
        # 第一层的权重和偏置使用键'W1'和'b1，第二层的权重和偏置使用键'W2'和'b2'。
        ############################################################################
        self.params["W1"] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params["b1"] = np.zeros(hidden_dim)
        self.params["W2"] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params["b2"] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        
        中文版本：
        计算一小批数据的损失和梯度。
        输入：
        - X：形状为(N, d_1, ..., d_k)的输入数据数组
        - y：形状为(N,)的标签数组。y[i]给出X[i]的标签。
        返回：
        如果y为None，则运行模型的测试时前向传递并返回：
        - scores：形状为(N, C)的数组，给出分类分数，其中scores[i, c]是X[i]和类c的分类分数。
        如果y不为None，则运行训练时的前向和后向传递，并返回一个元组：
        - loss：标量值，给出损失
        - grads：字典，具有与self.params相同的键，将参数名称映射到相对于这些参数的损失的梯度。
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        hidden, hidden_cache = affine_relu_forward(X, self.params["W1"], self.params["b1"])
        scores, scores_cache = affine_forward(hidden, self.params["W2"], self.params["b2"])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        # 中文版本：
        # 实现两层网络的反向传递。将损失存储在loss变量中，并将梯度存储在grads字典中。使用softmax计算数据损失，并确保grads[k]保存self.params[k]的梯度。不要忘记添加L2正则化！
        # 注意：为了确保您的实现与我们的实现匹配，并且您通过了自动化测试，请确保您的L2正则化包含一个0.5的因子，以简化梯度的表达式。
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (
            np.sum(self.params["W1"] * self.params["W1"])
            + np.sum(self.params["W2"] * self.params["W2"])
        )

        dhidden, grads["W2"], grads["b2"] = affine_backward(dscores, scores_cache)
        _, grads["W1"], grads["b1"] = affine_relu_backward(dhidden, hidden_cache)
        grads["W2"] += self.reg * self.params["W2"]
        grads["W1"] += self.reg * self.params["W1"]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      np.save(fpath, params) # type: ignore
      print(fname, "saved.")
    
    def load(self, fname):
      """Load model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True



class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    中文版本：
    多层全连接神经网络的类。
    网络包含任意数量的隐藏层、ReLU非线性和softmax损失函数。还将实现dropout和批量/层归一化作为选项。
    对于具有L层的网络，架构将是
    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
    其中批量/层归一化和dropout是可选的，并且{...}块重复L - 1次。
    可学习的参数存储在self.params字典中，并将使用Solver类进行学习。
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        中文版本：
        初始化一个新的FullyConnectedNet。
        输入：
        - hidden_dims：一个整数列表，给出每个隐藏层的大小。
        - input_dim：一个整数，给出输入的大小。
        - num_classes：一个整数，给出要分类的类别数。
        - dropout_keep_ratio：介于0和1之间的标量，给出dropout强度。如果dropout_keep_ratio=1，则网络根本不应使用dropout。
        - normalization：网络应使用的归一化类型。有效值为"batchnorm"、"layernorm"或None表示不进行归一化（默认）。
        - reg：标量，给出L2正则化强度。
        - weight_scale：标量，给出权重随机初始化的标准差。
        - dtype：一个numpy数据类型对象；所有计算都将使用此数据类型执行。float32更快但不太准确，因此您应该使用float64进行数值梯度检查。
        - seed：如果不为None，则将此随机种子传递给dropout层。这将使dropout层确定性，以便我们可以对模型进行梯度检查。
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        # 中文版本：
        # 初始化网络的参数，将所有值存储在self.params字典中。将第一层的权重和偏置存储在W1和b1中；对于第二层使用W2和b2，依此类推。权重应从以0为中心的正态分布初始化，标准差等于weight_scale。偏置应初始化为零。
        ############################################################################
        layer_dims = [input_dim] + list(hidden_dims) + [num_classes]
        for i in range(self.num_layers):
            self.params["W%d" % (i + 1)] = weight_scale * np.random.randn(
                layer_dims[i], layer_dims[i + 1]
            )
            self.params["b%d" % (i + 1)] = np.zeros(layer_dims[i + 1])
            if self.normalization in ("batchnorm", "layernorm") and i < self.num_layers - 1:
                self.params["gamma%d" % (i + 1)] = np.ones(layer_dims[i + 1])
                self.params["beta%d" % (i + 1)] = np.zeros(layer_dims[i + 1])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        # 中文版本：
        # 使用批量归一化时，我们需要跟踪运行均值和方差，因此我们需要将一个特殊的bn_param对象传递给每个批量归一化层。您应该将self.bn_params[0]传递给第一批归一化层的前向传递，将self.bn_params[1]传递给第二批归一化层的前向传递，依此类推。
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        中文版本：
        计算全连接网络的损失和梯度。
        输入：
        - X：形状为(N, d_1, ..., d_k)的输入数据数组
        - y：形状为(N,)的标签数组。y[i]给出X[i]的标签。
        返回：
        如果y为None，则运行模型的测试时前向传递并返回：
        - scores：形状为(N, C)的数组，给出分类分数，其中scores[i, c]是X[i]和类c的分类分数。
        如果y不为None，则运行训练时的前向和后向传递，并返回一个元组：
        - loss：标量值，给出损失
        - grads：与self.params具有相同键的字典，将参数名称映射到损失相对于这些参数的梯度。
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        # 中文版本：
        # 实现全连接网络的前向传递，计算X的类分数并将它们存储在scores变量中。
        # 使用dropout时，您需要将self.dropout_param传递给每个dropout前向传递。
        # 使用批量归一化时，您需要将self.bn_params[0]传递给第一批归一化层的前向传递，将self.bn_params[1]传递给第二批归一化层的前向传递，依此类推。
        ############################################################################
        caches = []
        dropout_caches = []
        out = X
        for i in range(1, self.num_layers):
            out, fc_cache = affine_forward(
                out, self.params["W%d" % i], self.params["b%d" % i]
            )
            norm_cache = None
            if self.normalization == "batchnorm":
                out, norm_cache = batchnorm_forward(
                    out,
                    self.params["gamma%d" % i],
                    self.params["beta%d" % i],
                    self.bn_params[i - 1],
                )
            elif self.normalization == "layernorm":
                out, norm_cache = layernorm_forward(
                    out,
                    self.params["gamma%d" % i],
                    self.params["beta%d" % i],
                    self.bn_params[i - 1],
                )
            out, relu_cache = relu_forward(out)
            drop_cache = None
            if self.use_dropout:
                out, drop_cache = dropout_forward(out, self.dropout_param)
            caches.append((fc_cache, norm_cache, relu_cache))
            dropout_caches.append(drop_cache)

        scores, final_cache = affine_forward(
            out,
            self.params["W%d" % self.num_layers],
            self.params["b%d" % self.num_layers],
        )
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        # 中文版本：
        # 实现全连接网络的反向传递。将损失存储在loss变量中，并将梯度存储在grads字典中。使用softmax计算数据损失，并确保grads[k]保存self.params[k]的梯度。不要忘记添加L2正则化！
        # 使用批量/层归一化时，您不需要对缩放和平移参数进行正则化。
        # 注意：为了确保您的实现与我们的实现匹配，并且您通过了自动化测试，请确保您的L2正则化包含一个0.5的因子，以简化梯度的表达式。
        ############################################################################
        loss, dout = softmax_loss(scores, y)
        for i in range(1, self.num_layers + 1):
            W = self.params["W%d" % i]
            loss += 0.5 * self.reg * np.sum(W * W)

        dout, grads["W%d" % self.num_layers], grads["b%d" % self.num_layers] = affine_backward(
            dout, final_cache
        )
        grads["W%d" % self.num_layers] += self.reg * self.params["W%d" % self.num_layers]

        for i in range(self.num_layers - 1, 0, -1):
            fc_cache, norm_cache, relu_cache = caches[i - 1]
            if self.use_dropout:
                dout = dropout_backward(dout, dropout_caches[i - 1])
            dout = relu_backward(dout, relu_cache)
            if self.normalization == "batchnorm":
                dout, grads["gamma%d" % i], grads["beta%d" % i] = batchnorm_backward_alt(
                    dout, norm_cache
                )
            elif self.normalization == "layernorm":
                dout, grads["gamma%d" % i], grads["beta%d" % i] = layernorm_backward(
                    dout, norm_cache
                )
            dout, grads["W%d" % i], grads["b%d" % i] = affine_backward(dout, fc_cache)
            grads["W%d" % i] += self.reg * self.params["W%d" % i]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


    def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      np.save(fpath, params) # type: ignore
      print(fname, "saved.")
    
    def load(self, fname):
      """Load model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True
