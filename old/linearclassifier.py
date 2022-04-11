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


class UnderstandClassifier:
    # weight matrix is by default np.ones
    def __init__(self, training_x, training_y, test_x, test_y, classes_length, hidden_layer_length=10):
        self.training_x = training_x
        self.training_y = training_y

        self.training_y_other_rep = np.zeros((self.training_x.shape[0], classes_length))
        for i in range(self.training_x.shape[0]):
            self.training_y_other_rep[i, self.training_y[i]] = 1

        self.test_x = test_x
        self.test_y = test_y
        self.W1 = np.ones((training_x.shape[1], hidden_layer_length))
        self.W2 = np.ones((hidden_layer_length, classes_length))

    def get_score(self, x_instance):
        return np.dot(np.maximum(0, np.dot(x_instance, self.W1)), self.W2)

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

    def cross_entropy_loss_function(self, x_instance, y_i):
        s = self.get_score(x_instance)
        return -np.log(np.e ** s[y_i] / np.sum(np.e ** s))

    def loss_function_for_the_dataset(self, x, y):
        n = len(x)
        loss_sum = 0.0
        for i in range(n):
            loss_sum += self.cross_entropy_loss_function(x[i], y[i])
        return loss_sum / n

    def loss_function_for_the_dataset_with_L2(self, x, y, regularization_strength=0.01):
        loss = self.loss_function_for_the_dataset(x, y)
        return loss + regularization_strength * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))

    # skip regularization
    def get_gradient(self):
        n = self.training_x.shape[0]

        # forward pass
        z1 = self.training_x.dot(self.W1)  # z1(120x10)
        z2 = np.maximum(0, z1)  # z2(120x10)
        z = z2.dot(self.W2)  # z(120x3)
        _S = z
        for i in range(n):
            _a = np.e ** z[i]
            _a_sum = _a.sum()
            for j in range(_S.shape[1]):
                _S[i, j] = _a[j]/_a_sum

        # backward pass
        dL_dz = (_S - self.training_y_other_rep)  # (120x3)

        dL_dW2 = z2.T @ dL_dz  # (10x3)
        dL_dz2 = dL_dz @ self.W2.T  # (120x10)

        dL_dz1 = np.empty(dL_dz2.shape)  # (120x10)
        for i in range(z1.shape[0]):
            for j in range(z1.shape[1]):
                if z1[i, j] > 0:
                    dL_dz1[i, j] = dL_dz2[i, j]
                else:
                    dL_dz1[i, j] = 0
        dL_dW1 = self.training_x.T @ dL_dz1
        return dL_dW1, dL_dW2

    def adam(self, learning_rate=0.001, beta1=0.9, beta2=0.999, iterations=50000, regularization_strength=0.001):
        self.W1 = np.random.normal(0, 1, size=self.W1.shape)
        self.W2 = np.random.normal(0, 1, size=self.W2.shape)
        moment11 = 0
        moment21 = 0
        moment12 = 0
        moment22 = 0
        regularization_strength *= 2
        for t in range(1, iterations + 1):
            dw1, dw2 = self.get_gradient()

            moment11 = beta1 * moment11 + (1 - beta1) * dw1
            moment21 = beta2 * moment21 + (1 - beta2) * (dw1 ** 2)
            moment11_unbias = moment11 / (1 - beta1 ** t)
            moment21_unbias = moment21 / (1 - beta2 ** t)

            moment12 = beta1 * moment12 + (1 - beta1) * dw2
            moment22 = beta2 * moment22 + (1 - beta2) * (dw2 ** 2)
            moment12_unbias = moment12 / (1 - beta1 ** t)
            moment22_unbias = moment22 / (1 - beta2 ** t)

            self.W1 -= learning_rate * (moment11_unbias / (np.sqrt(moment21_unbias) + 1e-7))
            self.W2 -= learning_rate * (moment12_unbias / (np.sqrt(moment22_unbias) + 1e-7))

            if t % 1000 == 0:
                print(t, 'ITERATIONS')
                accuracy, options = self.check_accuracy()
                print('loss:', self.loss_function_for_the_dataset(self.training_x, self.training_y), 'accuracy:', accuracy)
                print('chosen classes:', options)
        return self.loss_function_for_the_dataset_with_L2(self.training_x, self.training_y)


class BetterClassifier:
    # weight matrix is by default np.ones
    def __init__(self, training_x, training_y, test_x, test_y, classes_length, hidden_layer_length=10):
        self.training_x = training_x
        self.training_y = training_y
        self.test_x = test_x
        self.test_y = test_y
        self.W1 = np.ones((training_x.shape[1], hidden_layer_length))
        self.W2 = np.ones((hidden_layer_length, classes_length))

    def get_score(self, x_instance):
        return np.maximum(0, x_instance.dot(self.W1)).dot(self.W2)

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

    def cross_entropy_loss_function(self, x_instance, y_i):
        s = self.get_score(x_instance)
        return -np.log(np.e ** s[y_i] / np.sum(np.e ** s))

    def loss_function_for_the_dataset(self, x, y):
        n = len(x)
        loss_sum = 0.0
        for i in range(n):
            loss_sum += self.cross_entropy_loss_function(x[i], y[i])
        return loss_sum / n

    def loss_function_for_the_dataset_with_L2(self, x, y, regularization_strength=0.01):
        loss = self.loss_function_for_the_dataset(x, y)
        return loss + regularization_strength * np.sum(self.W1 ** 2) + regularization_strength * np.sum(self.W2 ** 2)

    def compute_gradient(self, h=0.001, regularization_strength=0.01):
        loss0 = self.loss_function_for_the_dataset_with_L2(self.training_x, self.training_y, regularization_strength=regularization_strength)
        grad1 = np.empty(self.W1.shape)
        grad2 = np.empty(self.W2.shape)
        for i in range(self.W1.shape[0]):
            for j in range(self.W1.shape[1]):
                self.W1[i, j] += h
                loss = self.loss_function_for_the_dataset_with_L2(self.training_x, self.training_y, regularization_strength=regularization_strength)
                grad1[i, j] = (loss - loss0) / h
                self.W1[i, j] -= h

        for i in range(self.W2.shape[0]):
            for j in range(self.W2.shape[1]):
                self.W2[i, j] += h
                loss = self.loss_function_for_the_dataset_with_L2(self.training_x, self.training_y, regularization_strength=regularization_strength)
                grad2[i, j] = (loss - loss0) / h
                self.W2[i, j] -= h
        return grad1, grad2

    # skip regularization
    def get_gradient(self):
        # forward pass
        x = self.training_x.T
        M1 = np.dot(self.W1, x)
        R1 = np.maximum(0, M1)
        M2 = np.dot(self.W2, R1)
        L = np.empty((M2.shape[1]))
        for i in range(M2.shape[1]):
            s = M2[:, i]
            L[i] = -np.log(np.e ** s[self.training_y[i]] / np.sum(np.e ** s))
        loss = np.mean(L)

    def adam(self, learning_rate=0.001, h=0.001, beta1=0.9, beta2=0.999, iterations=1000, regularization_strength=0.01):
        self.W1 = np.ones(self.W1.shape)  # init weight matrix
        self.W2 = np.ones(self.W2.shape)  # init weight matrix
        moment11 = 0
        moment21 = 0
        moment12 = 0
        moment22 = 0
        for t in range(1, iterations + 1):
            print('loss: ', self.loss_function_for_the_dataset_with_L2(self.training_x, self.training_y, regularization_strength=regularization_strength))
            dw1, dw2 = 1, 1, self.get_gradient()

            moment11 = beta1 * moment11 + (1 - beta1) * dw1
            moment21 = beta2 * moment21 + (1 - beta2) * (dw1 ** 2)
            moment11_unbias = moment11 / (1 - beta1 ** t)
            moment21_unbias = moment21 / (1 - beta2 ** t)

            moment12 = beta1 * moment12 + (1 - beta1) * dw2
            moment22 = beta2 * moment22 + (1 - beta2) * (dw2 ** 2)
            moment12_unbias = moment12 / (1 - beta1 ** t)
            moment22_unbias = moment22 / (1 - beta2 ** t)

            self.W1 -= learning_rate * moment11_unbias / (np.sqrt(moment21_unbias) + 1e-7)
            self.W2 -= learning_rate * moment12_unbias / (np.sqrt(moment22_unbias) + 1e-7)
        return self.W1, self.W2, self.loss_function_for_the_dataset_with_L2(self.training_x, self.training_y, regularization_strength=regularization_strength)
