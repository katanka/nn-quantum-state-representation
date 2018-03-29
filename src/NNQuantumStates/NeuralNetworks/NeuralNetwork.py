class NeuralNetwork(object):
    def __init__(self, display_name, X_train, Y_train):
        self.display_name = display_name
        self.X_train = X_train
        self.Y_train = Y_train

    def train(self):
        raise NotImplementedError()

    def predict(self, X_test):
        raise NotImplementedError()

    def validate(self, X_test, Y_test):
        raise NotImplementedError()