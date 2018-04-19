from nn_quantum_states.data.states import W

class Generator(object):
    def __init__(self, state):
        self.state = state

    def sample(self, num_samples):
        samples = []
        for _ in range(num_samples):
            samples.append(self.state.sample())

        return samples


if __name__ == "__main__":
    N = 5
    generator = Generator(W(5))
    generator.generate(5)