import numpy as np

def relu(x):
    """
    Compute the ReLU function.

    Parameters:
    x: np.ndarray 
        The input to the ReLU function
    
    Returns:
    out: np.ndarray
        The output of the ReLU function
    back: function to compute the derivative with respect to x
    """
    zeros = np.zeros_like(x)
    out = np.where(x > 0, x, zeros)
    back = lambda dout: dout * np.where(x >= 0, 1, 0) 
    return out, back

def linear_forward(x, W, b):
    """
    Compute the output of a linear layer.

    Parameters:
    x: np.ndarray of shape (batch_size, num_features)
        The input to the linear layer
    W: np.ndarray of shape (num_outputs, num_features)
        The weights of the linear layer
    b: np.ndarray of shape (num_outputs,)
        The bias of the linear layer

    Returns: 
    z: np.ndarray of shape (batch_size, num_outputs)
        The output of the linear layer
    back: function to compute the gradients with respect to x, W, and b using dz (batch_size, num_outputs)
    """
    z = x @ W.T + b 
    back = lambda dz: linear_backward(dz, x, W)
    return z, back

def linear_backward(dz, x, W):
    """
    Compute the gradients of a linear layer.

    Parameters:
    dz: np.ndarray of shape (batch_size, num_outputs)
        The partial derivative of the loss with respect to the output of the linear layer
    x: np.ndarray of shape (batch_size, num_features)
        The input to the linear layer
    w: np.ndarray of shape (num_outputs, num_features)
        The weights of the linear layer

    Returns:
    dx: np.ndarray of shape (batch_size, num_features)
        The partial derivative of the loss with respect to the input of the linear layer
    dw: np.ndarray of shape (num_outputs, num_features)
        The partial derivative of the loss with respect to the weights of the linear layer
    db: np.ndarray of shape (num_outputs,)
        The partial derivative of the loss with respect to the bias of the linear layer
    """
    dx = np.dot(dz, W) # layer output changes wrt input according to **weight**
    dw = np.dot(dz.T, x) # layer output changes wrt weight according to **input**. transpose needed, though
    db = np.sum(dz) # layer output does not change wrt bias. that's the point of bias.
    return dx, dw, db

class DenseLayer(object):
    def __init__(self, num_inputs, num_outputs, activation=None):
        """
        Initialize the dense layer.
        """
        # each weight matrix entry should be sampled from a normal distribution with mean 0 
        # and stdev 1/(sqrt(num_inputs)). this can be done easily with np.random.normal((outs, ins))
        # which yields mean 0 and unit stdev; this can be easily scaled by multiplying by the desired 
        # 1/(sqrt(ins)).
        self.W = np.random.randn(num_outputs, num_inputs) * (2.0/(np.sqrt(num_inputs)))
        self.b = np.zeros(num_outputs)  # initialize the bias to 0
        self.activation = activation
    
    def parameters(self):
        """
        Return the parameters of the model.
        """
        return self.W, self.b
    
    def forward(self, x):
        """
        Compute the output of the linear layer.
        """
        z, linear_back = linear_forward(x, self.W, self.b)
        if self.activation is None:
            return z, linear_back
        else:
            h, act_back = self.activation(z)
            back = lambda dh: linear_back(act_back(dh))
            return h, back


class NN(object):
    def __init__(self, num_inputs, hidden_sizes, hidden_activation, num_outputs):
        """
        Initialize the network.

        Parameters:
        num_inputs: int
            The number of input features
        hidden_sizes: list of int
            The number of hidden units in each hidden layer
        hidden_activation: function
            The activation function to use in the hidden layers
        num_outputs: int
            The number of output units
        """
        self.layers = []
        for i in range(len(hidden_sizes)):
            self.layers.append(DenseLayer(num_inputs, hidden_sizes[i], hidden_activation))
            num_inputs = hidden_sizes[i]
        self.layers.append(DenseLayer(num_inputs, num_outputs, None))
        
    def forward(self, x):
        """
        Compute the output of the network and return a function to compute the gradients.
        """
        backs = []
        for layer in self.layers:
            x, back = layer.forward(x)
            backs.append(back)
        
        def net_back(dh):
            grads = {}
            i = len(self.layers) - 1
            for back in reversed(backs):
                dx, dw, db = back(dh)
                # grads are stored in a dictionary with the same key as the parameters
                grads[f'W{i}'] = dw  
                grads[f'b{i}'] = db
                dh = dx
                i -= 1
            return grads

        return x, net_back
    
    def parameters(self):
        """
        Return the parameters of the model.
        """
        params = {}
        for i, layer in enumerate(self.layers):
            params[f'W{i}'], params[f'b{i}'] = layer.parameters()
        return params
    
def cross_entropy_loss(h, y):
    """
    Compute the cross-entropy loss and the gradient of the loss with respect to h.

    Parameters:
    y_hat: np.ndarray of shape (batch_size, num_classes)
        The output of the model before the softmax function
    y: np.ndarray of shape (batch_size, num_classes)
        The true class probabilities

    Returns:
    loss: float
        The cross-entropy loss
    dy_hat: np.ndarray of shape (batch_size, num_classes)
        The gradient of the loss with respect to y_hat
    """
    def safe_log(x):
        return np.log(np.where(x != 0, x, 1e-8))
    
    m = h.shape[0]
    softmax_output = softmax(h)
    loss = -np.sum(y * safe_log(softmax_output)) / m
    dh = (softmax_output - y) / m
    back = lambda dl: dl * dh

    return loss, back

def softmax(h):
    """
    Compute the softmax function. The return represents the probablity of each class. 

    Parameters:
    h: np.ndarray of shape (batch_size, num_classes)
        The input to the softmax function

    Returns:
    out: np.ndarray of shape (batch_size, num_classes)
        The output of the softmax function
    """
    out = np.exp(h) / np.sum(np.exp(h), axis=-1, keepdims=True)
    return out

def try_nn():
    X = np.random.randn(10, 5)
    y = np.zeros((10, 2))
    y[np.arange(10), np.random.randint(0, 2, 10)] = 1
    model = NN(5, [10, 10], relu, 2)
    h, back = model.forward(X)
    loss, dl = cross_entropy_loss(h, y)
    grads = back(dl(1))
    print(loss)
    print(grads)
    params = model.parameters()
    for key in params:
        print(key, params[key].shape, grads[key].shape)

def finite_difference(f, arg, eps=1e-6):
    """
    Compute the finite difference of a function. 

    Parameters:
    f: function
        The function to compute the finite difference of
    args: np.ndarray
        The arguments to the function
    eps: float
        The epsilon to use in the finite difference computation
    num_checks: int
        The number of finite difference checks to perform

    Returns:
    grad_approx: list of np.ndarray
        The average difference between the finite difference and the true gradient
    """
    grad_approx = np.zeros_like(arg)
    for i in range(arg.size):
        arg.flat[i] += eps
        f1 = f(arg)
        arg.flat[i] -= 2 * eps
        f2 = f(arg)
        arg.flat[i] += eps
        grad_approx.flat[i] = (f1 - f2) / (2 * eps)
    return grad_approx            

def test_relu():
    # test derivative of relu 
    h, back = relu(-1)
    assert h == 0
    assert back(1) == 0
    h, back = relu(1)
    assert h == 1
    assert back(1) == 1
    h, back = relu(0)
    assert h == 0
    assert back(1) == 1

    # check for array input
    x = np.array([[-1, 0, 2]])
    h, back = relu(x)
    assert np.allclose(h, [[0, 0, 2]])
    dh = np.array([[1, 1, 1]])
    dx = back(dh)
    assert np.allclose(dx, [[0, 1, 1]])
    print("All `relu` tests pass.")

def test_forward():
    np.random.seed(0)
    x = np.random.randn(1, 5)
    w = np.random.randn(2, 5) / np.sqrt(5)
    b = np.random.randn(2) / np.sqrt(5)
    z, back = linear_forward(x, w, b)
    # test derivative for only single output of layer
    for i in range(w.shape[0]):
        dz = np.zeros_like(z)
        dz[0,i] = 1
        dx, dw, db = back(dz)
        dw_approx = finite_difference(lambda w: linear_forward(x, w, b)[0][0,i], w, eps=1e-8)                                        
        db_approx = finite_difference(lambda b: linear_forward(x, w, b)[0][0,i], b, eps=1e-8)                                        
        dx_approx = finite_difference(lambda x: linear_forward(x, w, b)[0][0,i], x, eps=1e-8)
        # uncomment these to see the derivatives
        # print("dw", dw, "\n", dw_approx)
        # print("db", db, "\n", db_approx)
        # print("dx", dx, "\n", dx_approx)
        assert np.allclose(dw, dw_approx, atol=1e-4), f"dw: {dw}\ndw_approx: {dw_approx}"
        assert np.allclose(db, db_approx, atol=1e-4), f"db: {db}\ndb_approx: {db_approx}"
        assert np.allclose(dx, dx_approx, atol=1e-4), f"dx: {dx}\ndx_approx: {dx_approx}"
    print("All `forward pass` tests pass.")

def test_loss():
    # test derivative of cross entropy loss for 2 classes
    h = np.array([[0.0, 1.0]]) # pre softmax
    y = np.array([[0, 1]]) # 2nd class is correct
    p = np.exp(h) / np.sum(np.exp(h), axis=-1, keepdims=True)
    loss, back = cross_entropy_loss(h, y)
    probs = np.exp(h) / np.sum(np.exp(h), axis=-1, keepdims=True)
    assert np.allclose(loss, -np.log(probs[0,1]))  # check that the loss is correct
    dh = back(1)
    dh_approx = finite_difference(lambda h: cross_entropy_loss(h, y)[0], h, eps=1e-4)
    assert np.allclose(dh, dh_approx, atol=1e-4), f"\nback: {back(1)}, dh_approx: {dh_approx}"
    
    # test for 3 classes
    h = np.array([[0, 1.0, 2.0]])
    y = np.array([[0, 1, 0]])  # 2nd class is correct
    loss, back = cross_entropy_loss(h, y)
    dh = back(1)
    dh_approx = finite_difference(lambda h: cross_entropy_loss(h, y)[0], h, eps=1e-8)
    assert np.allclose(dh, dh_approx, atol=1e-4), f"\nback: {back(1)}, dh_approx: {dh_approx}"

    # check for 2 examples:
    h = np.array([[0.0, 1.0], [1.0, 0.0]]) # pre softmax
    y = np.array([[0, 1], [1, 0]]) # 2nd class is correct
    loss, back = cross_entropy_loss(h, y)
    probs = np.exp(h) / np.sum(np.exp(h), axis=-1, keepdims=True)
    dh = back(1)
    assert np.allclose(loss, -(np.log(probs[0,1]) + np.log(probs[1,0]))/2), f"loss: {loss}, probs: {probs}"
    dh_approx = finite_difference(lambda h: cross_entropy_loss(h, y)[0], h, eps=1e-8)
    assert np.allclose(back(1), dh_approx, atol=1e-4), f"back: {back(1)}, dh_approx: {dh_approx}"

    print("All `loss` tests pass.")
    
if __name__ == '__main__':
    try_nn()
    test_loss()
    test_relu()
    test_forward() 