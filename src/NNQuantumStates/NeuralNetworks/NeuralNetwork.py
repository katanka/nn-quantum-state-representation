class NeuralNetwork(object):
    def __init__(self, display_name, num_nodes_in_layer, num_layers):
        self.display_name = display_name
        self.num_layers = num_layers
        self.num_nodes_in_layer = np.zeros((1, num_layers))

    def train(self):
        raise NotImplementedError()

    def predict(self, X_test):
        raise NotImplementedError()

    def validate(self, X_test, Y_test):
        raise NotImplementedError()