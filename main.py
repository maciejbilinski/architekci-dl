import numpy as np
from csvreader import CSVReader
from datamanager import DataManager
from linearclassifier import LinearClassifier, cross_entropy_loss_function, loss_function_for_the_dataset, \
    loss_function_for_the_dataset_with_L2

iris = CSVReader('data/IRIS.csv')

print('RAW DATA')
print(iris.get_data())

print('RAW DATA AS NUMPY')
data = iris.get_numpy()
print(data)

manager = DataManager(data)
print("SEPARATED X AND Y")
x, y = manager.separate_x_y()
print('X')
print(x)
print('Y')
print(y)

print('CONVERTED X TO FLOATS')
x = manager.convert_x_to_floats()
print(x)

print('CONVERTED CLASSES NAMES TO IDS')
convention, y = manager.convert_classes_to_numbers()
print(y)
print(convention)
classes = [x[1] for x in convention]

print('BIAS TRICK')
x = manager.add_bias_trick()
print(x)

print('TRAINING AND TEST SETS')
training_x, training_y, test_x, test_y = manager.get_training_test_set_in_equal_portion()
print('TRAINING SET')
print(training_x, training_y)
print('TEST SET (20% OF THE DATASET)')
print(test_x, test_y)

print('LINEAR CLASSIFIER')
classifier = LinearClassifier(training_x, training_y, test_x, test_y, manager.get_classes_length())

print('TEST #1')
print('BY DEFAULT WEIGHT MATRIX IS NUMPY ONES, SO EVERY CLASS SHOULD HAVE THE SAME SCORE')
print('INSTANCE:', training_x[0])
print('SCORE:', classifier.get_score(training_x[0]))

print('TEST #2')
print('CHECK MATH OPERATIONS OF THE CROSS-ENTROPY LOSS FUNCTION')
cross_entropy_loss_function(training_x[0], classifier.W, training_y[0], debug=True)

print('TEST #3')
print('CHECK LOSS FUNCTION FOR THE DATASET')
print('LOSS:', loss_function_for_the_dataset(training_x, classifier.W, training_y))

print('TEST #4')
print('REGULARIZATION L2')
print('CHECK LOSS WILL BE SMALLER FOR MORE \'SPREAD OUT\' WEIGHT MATRIX')
w1 = np.zeros(classifier.W.shape)
w1[:, 0] = 1.0
print('W:', w1)
print('LOSS:', loss_function_for_the_dataset(training_x, w1, training_y))
print('LOSS L2:', loss_function_for_the_dataset_with_L2(training_x, w1, training_y))
w2 = np.full(classifier.W.shape, 0.25)
print('W:', w2)
print('LOSS:', loss_function_for_the_dataset(training_x, w2, training_y))
print('LOSS L2:', loss_function_for_the_dataset_with_L2(training_x, w2, training_y))

print("TEST #5")
print('NAIVE RANDOM WEIGHT MATRIX SEARCH')
print('ALWAYS STARTS WITH NUMPY ONES WEIGHT MATRIX')
for i in range(200, 1200, 200):
    print(i, 'ITERATIONS')
    w, loss = classifier.random_search(iterations=i)
    accuracy, options = classifier.check_accuracy()
    print('loss:', loss, 'accuracy:', accuracy)
    print('chosen classes:', options)
log_c = np.log(manager.get_classes_length())
print('LOG(C):', log_c)

print('TEST #6')
print('GRADIENT DESCENT WITH NUMERIC GRADIENT')
print('1000 ITERATIONS')
w, loss = classifier.gradient_descent()
accuracy, options = classifier.check_accuracy()
print('loss:', loss, 'accuracy:', accuracy)
print('chosen classes:', options)

print('TEST #7')
print('ADAM WITH NUMERIC GRADIENT')
print('1000 ITERATIONS')
w, loss = classifier.adam()
accuracy, options = classifier.check_accuracy()
print('loss:', loss, 'accuracy:', accuracy)
print('chosen classes:', options)

