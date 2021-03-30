import numpy as np
from BoltzmannMachine.boltzmann_machine import BoltzmannMachine
from ArrowDataset import ArrowDataset

import matplotlib.pyplot as plt

X = 5
Y = 5
S = 1

H = 128


def collect_distribution_samples(model, dataset):
    return [model.gibbs_sampling(visible_state=dataset[i].reshape(-1)) for i in range(len(dataset))]

def collect_model_samples(model, n_data_samples=10, n_gibbs_samples=10):
    return [model.gibbs_sampling(iters=n_gibbs_samples) for _ in range(n_data_samples)]

def compute_gradient(distribution_samples, model_samples, model):
    size = model.total_size

    d_gradient_w = np.zeros((size, size))
    d_gradient_b = np.zeros(size)
    m_gradient_w = np.zeros((size, size))
    m_gradient_b = np.zeros(size)

    for samp in distribution_samples:
        d_gradient_w += np.outer(samp, samp)
        d_gradient_b += samp
    d_gradient_w = d_gradient_w / len(distribution_samples)
    d_gradient_b = d_gradient_b / len(distribution_samples)

    for samp in model_samples:
        m_gradient_w += np.outer(samp, samp)
        m_gradient_b += samp
    m_gradient_w = m_gradient_w / len(model_samples)
    m_gradient_b = m_gradient_b / len(model_samples)

    dw = d_gradient_w - m_gradient_w
    db = d_gradient_b - m_gradient_b

    return dw, db


def run_train(lr=0.05, epochs=10, n_data_samples=10, n_gibbs_samples=10, collection_timesteps=None):
    boltzmann_machine = BoltzmannMachine(X * Y, H)
    ds = ArrowDataset(Y, X, S, n_data_samples)

    if collection_timesteps is None:
        collected_samples = None
        collection_timesteps = []
    else:
        collected_samples = []

    for e in range(epochs):
        d_samps = collect_distribution_samples(boltzmann_machine, ds)
        m_samps = collect_model_samples(boltzmann_machine, n_data_samples=n_data_samples, n_gibbs_samples=n_gibbs_samples)
        dw, db = compute_gradient(d_samps, m_samps, boltzmann_machine)
        boltzmann_machine.update_params_from_gradient(dw, db, lr)
        print(f"\rEpoch {e + 1}/{epochs}", end='')

        if e + 1 in collection_timesteps:
            collected_samples.append([boltzmann_machine.gibbs_sampling()[:X*Y].reshape((Y, X)) for _ in range(3)])

    return collected_samples
if __name__ == "__main__":
    collection_timesteps = [1, 200, 500]
    samples = run_train(lr=0.05, epochs=500, n_data_samples=50, n_gibbs_samples=20, collection_timesteps=collection_timesteps)

    for group, epoch in zip(samples, collection_timesteps):
        for i, sample in enumerate(group, 1):
            plt.subplot(1, 3, i)
            plt.imshow(sample)
        plt.suptitle(f"Generated samples from BM at epoch {epoch}")
        plt.show()