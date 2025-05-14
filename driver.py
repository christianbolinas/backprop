import numpy as np
import matplotlib.pyplot as plt

from nn import relu, NN, cross_entropy_loss

class BatchGenerator(object):
    def __init__(self, x, y, batch_size):
        """
        Initialize the batch generator object.

        Parameters:
        x: np.ndarray of shape (num_samples, num_features)
            The input features
        y: np.ndarray of shape (num_samples, num_outputs)
            The output labels
        batch_size: int
            The size of each mini-batch
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.num_samples = x.shape[0]
        self.num_batches = self.num_samples // self.batch_size

    def __iter__(self):
        """
        Return the iterator object.
        """
        self.current_batch = 0
        return self
    
    def __next__(self):
        """
        Return the next batch.
        """
        if self.current_batch >= self.num_batches:
            raise StopIteration
        # use self.current_batch to compute start and end points
        # the start index is initialized to 0, and upon iteration is incremented by the batch size
        # the end index is a `batch size` after that
        # and the batches are the slices of the training (features/label) pairs
        start = self.current_batch * self.batch_size
        end = start + self.batch_size
        self.current_batch += 1
        return self.x[start:end], self.y[start:end]
    
    def __len__(self):
        """
        Return the number of batches.
        """
        return self.num_batches
    
    def shuffle(self):
        """
        Randomly shuffle the data.
        """
        # it's imperative that the training features and training labels are shuffled TOGETHER,
        # so a naive `x, y = shuffled(x), shuffled(y)` will not do. The following is adapted from
        # https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
        perm = np.random.permutation(len(self.x))
        self.x = self.x[perm]
        self.y = self.y[perm]

class Adam(object):
    def __init__(self, eta, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Initialize the Adam optimizer.

        Parameters:
        eta: float
            The learning rate
        beta1: float
            The exponential decay rate for the first moment estimates
        beta2: float
            The exponential decay rate for the second moment estimates
        eps: float
            A small constant to prevent division by zero
        """
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}
    
    def update(self, params, grads):
        """
        Update the parameters using the Adam optimizer.

        Parameters:
        params: dict
            The parameters of the model
        grads: dict
            The gradients of the parameters with respect to the loss. They have the same key as the parameters.
        """
        self.t += 1
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])  # initialize parameters for the moving average of the gradient
                self.v[key] = np.zeros_like(params[key])  # initialize parameters for the moving average of the squared gradient

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key] ** 2

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            params[key] -= self.eta * m_hat / (np.sqrt(v_hat) + self.eps) 

def train(model, data, optimizer, num_epochs=100, call_back=None):
    """
    Train the model using the given data.

    Parameters:
    model: NN
        The neural network model
    data: BatchGenerator
        The data generator
    num_epochs: int
        The number of epochs
    call_back: function
        The callback function to call after each epoch
    """
    losses = []
    accs = []  # list to store the accuracy of the model
    for epoch in range(num_epochs):
        data.shuffle()
        for x_batch, y_batch in data:
            h, back = model.forward(x_batch)

            loss, dl = cross_entropy_loss(h, y_batch)

            losses.append(loss)
            acc = np.mean(np.argmax(h, axis=1) == np.argmax(y_batch, axis=1))  # computes accuracy best on argmax output of the network
            accs.append(acc)
            grads = back(dl(1))  # computes the grads of the loss with respect to the parameters
            optimizer.update(model.parameters(), grads)  # updates the parameters of the model
        print(f'Epoch {epoch + 1}: Train loss = {loss}')
        if call_back is not None:
            call_back(model, epoch)  # do any logging or model evaluation in the callback
    return losses, accs

def load_dataset():
    """
    Load the dataset.
    """
    # Load the MNIST dataset from disk
    X_tr, Y_tr = np.load('x_train.npy'), np.load('y_train.npy')
    X_te, Y_te = np.load('x_test.npy'), np.load('y_test.npy')
    return X_tr, Y_tr, X_te, Y_te

def make_one_hot(y, num_classes):
    """
    Convert the labels to one-hot encoding.

    Parameters:
    y: np.ndarray of shape (num_samples,)
        The labels
    num_classes: int
        The number of classes

    Returns:
    np.ndarray of shape (num_samples, num_classes)
        The one-hot encoding of the labels
    """
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot

if __name__ == '__main__':
    np.random.seed(0) 
    X_tr, Y_tr, X_te, Y_te = load_dataset()
    Y_tr = make_one_hot(Y_tr, 10)
    Y_te = make_one_hot(Y_te, 10)
    
    # THESE ARE THE HYPERPARAMETERS
    batch_size = 64 
    hidden_sizes = [512, 256]
    step_size = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    num_epochs = 15 

    data = BatchGenerator(X_tr, Y_tr, batch_size=batch_size)

    model = NN(784, hidden_sizes=hidden_sizes, hidden_activation=relu, num_outputs=10)
    optimizer = Adam(step_size, beta1=beta1, beta2=beta2, eps=eps)
    test_losses = []
    test_accuracy = []
    def call_back(model, epoch):
        h, _ = model.forward(X_te)
        loss, _ = cross_entropy_loss(h, Y_te)
        print(f'Epoch {epoch+1}: Test loss = {loss}')
        test_losses.append(loss)
        test_accuracy.append(np.mean(np.argmax(h, axis=1) == np.argmax(Y_te, axis=1)))

    losses, accs = train(model, data, optimizer, num_epochs=num_epochs, call_back=call_back)

    # Plot the loss and accuracy
    losses_per_epoch = [losses[i:i + len(data)] for i in range(0, len(losses), len(data))]
    accs_per_epoch = [accs[i:i + len(data)] for i in range(0, len(accs), len(data))]
    fig, axs = plt.subplots(2,1, figsize=(6, 6))
    for i, loss in enumerate(losses_per_epoch):
        axs[0].scatter([i]*len(loss), loss, color="dodgerblue", s=2, alpha=0.5, label=None)
    for i, acc in enumerate(accs_per_epoch):
        axs[1].scatter([i]*len(acc), acc, color="dodgerblue", s=2, alpha=0.5, label=None)
    losses = [np.mean(loss) for loss in losses_per_epoch]
    accs = [np.mean(acc) for acc in accs_per_epoch]
    axs[0].plot(losses, color="dodgerblue", label="Train")
    axs[0].plot(test_losses, color="crimson", label="Test")
    axs[0].set_ylabel('Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_yscale('log')
    axs[0].legend()
    axs[1].plot(accs, color="dodgerblue")
    axs[1].plot(test_accuracy, color="crimson")
    axs[1].set_ylabel('Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylim(0.8, 1)
    print(f"Final test accuracy: {test_accuracy[-1]}")
    plt.savefig('loss_acc.pdf')