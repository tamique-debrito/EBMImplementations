import numpy as np
from BoltzmannMachine.restricted_boltzmann_machine import RestrictedBoltzmannMachine
from ArrowDataset import ArrowDataset

import matplotlib.pyplot as plt

X = 7
Y = 7
S = 2

H = 16


def collect_distribution_samples(model, dataset):
    return [model.gibbs_sampling(visible_state=dataset[i].reshape(-1)) for i in range(len(dataset))]

def collect_model_samples(model, n_data_samples=10, n_gibbs_samples=10):
    return [model.gibbs_sampling(iters=n_gibbs_samples) for _ in range(n_data_samples)]

def compute_gradient(distribution_samples, model_samples, model):
    v = model.visible_size
    h = model.hidden_size

    d_gradient_w = np.zeros((v, h))
    d_gradient_b_v = np.zeros(v)
    d_gradient_b_h = np.zeros(h)
    m_gradient_w = np.zeros((v, h))
    m_gradient_b_v = np.zeros(v)
    m_gradient_b_h = np.zeros(h)

    for samp_v, samp_h in distribution_samples:
        d_gradient_w += np.outer(samp_v, samp_h)
        d_gradient_b_v += samp_v
        d_gradient_b_h += samp_h
    d_gradient_w = d_gradient_w / len(distribution_samples)
    d_gradient_b_v = d_gradient_b_v / len(distribution_samples)
    d_gradient_b_h = d_gradient_b_h / len(distribution_samples)

    for samp_v, samp_h in model_samples:
        m_gradient_w += np.outer(samp_v, samp_h)
        m_gradient_b_v += samp_v
        m_gradient_b_h += samp_h
    m_gradient_w = m_gradient_w / len(model_samples)
    m_gradient_b_v = m_gradient_b_v / len(model_samples)
    m_gradient_b_h = m_gradient_b_h / len(model_samples)

    dw = d_gradient_w - m_gradient_w
    db_v = d_gradient_b_v - m_gradient_b_v
    db_h = d_gradient_b_h - m_gradient_b_h

    return dw, db_v, db_h


def run_train(lr=0.1, epochs=10, n_data_samples=10, n_gibbs_samples=10, collection_timesteps=None):
    boltzmann_machine = RestrictedBoltzmannMachine(X * Y, H)
    ds = ArrowDataset(Y, X, S, n_data_samples)

    if collection_timesteps is None:
        collected_samples = None
        collection_timesteps = []
    else:
        collected_samples = []

    for e in range(epochs):
        d_samps = collect_distribution_samples(boltzmann_machine, ds)
        m_samps = collect_model_samples(boltzmann_machine, n_data_samples=n_data_samples, n_gibbs_samples=n_gibbs_samples)
        dw, db_v, db_h = compute_gradient(d_samps, m_samps, boltzmann_machine)
        boltzmann_machine.update_params_from_gradient(dw, db_v, db_h, lr)
        print(f"\rEpoch {e + 1}/{epochs}", end='')

        if e + 1 in collection_timesteps:
            collected_samples.append([boltzmann_machine.gibbs_sampling()[0].reshape((Y, X)) for _ in range(3)])

    return collected_samples

def show_arrows():
    ds = ArrowDataset(Y, X, S, 0)
    for i in range(1, 4):
        plt.subplot(1, 3, i)
        plt.imshow(ds[0])
    plt.show()

if __name__ == "__main__":
    #show_arrows()
    collection_timesteps = [1, 200, 500]
    samples = run_train(lr=0.05, epochs=500, n_data_samples=50, n_gibbs_samples=20, collection_timesteps=collection_timesteps)

    for group, epoch in zip(samples, collection_timesteps):
        for i, sample in enumerate(group, 1):
            plt.subplot(1, 3, i)
            plt.imshow(sample)
        plt.suptitle(f"Samples from RBM at epoch {epoch}")
        plt.show()