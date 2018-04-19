from nn_quantum_states.networks.rbm import RBM
from nn_quantum_states.data.generator import Generator
from nn_quantum_states.data.states import W

num_spins = 5
alpha = 1

num_training_samples = 50

rbm = RBM(num_spins, alpha*num_spins)
true_state = W(num_spins)
generator = Generator(true_state)
training_samples = generator.sample(num_training_samples)


Trainer(rbm, training_samples).train()
