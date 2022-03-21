import numpy as np


def _get_score(x_instance, W):
    return np.dot(W, x_instance)


def cross_entropy_loss_function(x_instance, W, y_i, debug=False):
    s = _get_score(x_instance, W)
    if debug:
        print('SCORE:', s)
        e_s = np.e ** s
        print('e^s: ', e_s)
        e_s_sum = np.sum(e_s)
        print('sum of the e^s:', e_s_sum)
        normalized_probability = np.e ** s[y_i] / e_s_sum
        print('normalized probability:', normalized_probability)
        result = -np.log(normalized_probability)
        print('loss:', result)
        return result
    return -np.log(np.e ** s[y_i] / np.sum(np.e ** s))


def loss_function_for_the_dataset(x, W, y, loss_fn=cross_entropy_loss_function):
    n = len(x)
    loss_sum = 0.0
    for i in range(n):
        loss_sum += loss_fn(x[i], W, y[i])
    return loss_sum/n


def loss_function_for_the_dataset_with_L2(x, W, y, loss_fn=cross_entropy_loss_function, regularization_strength=0.01):
    loss = loss_function_for_the_dataset(x, W, y, loss_fn)
    return loss + regularization_strength * np.sum(W**2)


# numerical gradient
def compute_gradient(training_x, training_y, W, h=0.001, regularization_strength=0.01):
    loss0 = loss_function_for_the_dataset_with_L2(training_x, W, training_y, regularization_strength=regularization_strength)
    grad = np.empty(W.shape)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] += h
            loss = loss_function_for_the_dataset_with_L2(training_x, W, training_y, regularization_strength=regularization_strength)
            grad[i, j] = (loss - loss0)/h
            W[i, j] -= h
    return grad


class LinearClassifier:
    # weight matrix is by default np.ones
    def __init__(self, training_x, training_y, test_x, test_y, classes_length):
        self.training_x = training_x
        self.training_y = training_y
        self.test_x = test_x
        self.test_y = test_y
        self.W = np.ones((classes_length, training_x.shape[1]))

    def get_score(self, x_instance):
        return _get_score(x_instance, self.W)

    def random_search(self, iterations=1000, regularization_strength=0.01):
        self.W = np.ones(self.W.shape)
        best_loss = loss_function_for_the_dataset_with_L2(self.training_x, self.W, self.training_y, regularization_strength=regularization_strength)
        for num in range(0, iterations):
            weight_matrix = np.random.randn(*self.W.shape) * 0.001  # from Michigan Online
            loss = loss_function_for_the_dataset_with_L2(self.training_x, weight_matrix, self.training_y, regularization_strength=regularization_strength)
            if loss < best_loss:
                best_loss = loss
                self.W = weight_matrix
        return self.W, best_loss

    def check_accuracy(self):
        correct = 0
        options = [0, 0, 0]
        for i in range(0, self.test_x.shape[0]):
            result = self.get_score(self.test_x[i])
            selected_class = np.argmax(result)
            options[selected_class] += 1
            if selected_class == self.test_y[i]:
                correct = correct + 1

        return correct / self.test_x.shape[0], options

    def gradient_descent(self, learning_rate=0.01, h=0.001, iterations=1000, regularization_strength=0.01):
        self.W = np.ones(self.W.shape)  # init weight matrix
        for i in range(iterations):
            grad = compute_gradient(self.training_x, self.training_y, self.W, h, regularization_strength)
            self.W -= learning_rate * grad
        return self.W, loss_function_for_the_dataset_with_L2(self.training_x, self.W, self.training_y, regularization_strength=regularization_strength)

    def adam(self, learning_rate=0.001, h=0.001, beta1=0.9, beta2=0.999, iterations=1000, regularization_strength=0.01):
        self.W = np.ones(self.W.shape)  # init weight matrix
        moment1 = 0
        moment2 = 0
        for t in range(1, iterations + 1):
            dw = compute_gradient(self.training_x, self.training_y, self.W, h, regularization_strength)
            moment1 = beta1 * moment1 + (1 - beta1) * dw
            moment2 = beta2 * moment2 + (1 - beta2) * (dw ** 2)
            moment1_unbias = moment1 / (1 - beta1 ** t)
            moment2_unbias = moment2 / (1 - beta2 ** t)
            self.W -= learning_rate * moment1_unbias / (np.sqrt(moment2_unbias) + 1e-7)
        return self.W, loss_function_for_the_dataset_with_L2(self.training_x, self.W, self.training_y, regularization_strength=regularization_strength)
