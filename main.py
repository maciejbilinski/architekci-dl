import irisnetwork

network = irisnetwork.IrisNetwork()
loss = network.train(1000)
accuracy, options = network.check_accuracy()

print('training loss:', loss)
print('accuracy:', accuracy)
print('chosen classes:', options)
