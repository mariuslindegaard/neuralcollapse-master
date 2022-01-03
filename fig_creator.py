import matplotlib.pyplot as plt
import numpy as np
import torchsummary as ths
import torch as th
import pickle
import sys
import os

src_path = os.path.join(os.getcwd(), 'src')
if not src_path in sys.path:
    sys.path.append(src_path)

import our_models as om

# modelfc = om.NetSimpleConv2FC(3, 32, 10, use_softmax=True)
# model = om.NetSimpleConv(3, 32, 10)


from do_measurements import Measurements
def get_measurements(dirname):
    with open(os.path.join('runs', dirname, 'measurements/measurements.pkl'), 'rb') as f:
        return pickle.load(f)
def get_measurements_stl(dirname):
    with open(os.path.join('runs', dirname, 'stl_measurements/measurements.pkl'), 'rb') as f:
        return pickle.load(f)

def get_both(dirname):
    return get_measurements(dirname), get_measurements_stl(dirname)


metrics = ('accuracy', 'loss', 'reg_loss', 'Sw_invSb', 'norm_M_CoV', 'norm_W_CoV', 'cos_M', 'cos_W',
           'W_M_dist', 'NCC_mismatch',
           'SQI_eps1_avg', 'SQI_eps2_avg', 'SQI_eps1_over_C-1_eps2', 'SQI_eps1_rel_std', 'SQI_eps2_rel_std',
           'SQI_eps2_sample_rel_std')

epochs = [1,   2,   3,   4,   5,   6,   7,   8,   9,   10,   11,
               12,  13,  14,  16,  17,  19,  20,  22,  24,  27,   29,
               32,  35,  38,  42,  45,  50,  54,  59,  65,  71,   77,
               85,  92,  101, 110, 121, 132, 144, 158, 172, 188,  206,
               225, 245, 268, 293, 320, 350, 400, 450, 500, 550, 600,
               650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100]

def plot_eps1_2():
    c1fc =  get_measurements('cifar_1fc')
    c2fc, c2fc_stl = get_both('cifar_2fc')
    m2fc = get_measurements('mnist_2fc')

    plt.figure(figsize=(16, 6))

    runs = [(c1fc, 'r', 'cifar10, 1FC'),
            (c2fc, 'b', 'cifar10, 2FC'),
            (m2fc, 'k', 'mnist, 2FC')]
    subplots = [((1, 2, 1), r"Average $\epsilon_{i, c}$ and $(C-1)\tilde\epsilon_{i, c, c'}$"),
                ((1, 2, 2), r'Empirical relative standard-deviation')]
    measures = [('SQI_eps1_avg', 'o-', 0, '$\epsilon^{avg}$, '),
                ('SQI_eps2_avg', 'x--', 0, r'$(C-1)\tilde\epsilon^{avg}$'),
                ('SQI_eps1_rel_std', 'o-', 1, r'std$(\epsilon / \epsilon^{avg})$'),
                ('SQI_eps2_rel_std', 'x--', 1, r'std$(\tilde\epsilon / \tilde\epsilon^{avg})$')]
    for m_name, linetype, subplot, label_1 in measures:
        ax = plt.subplot(*subplots[subplot][0])
        for run_measurements, run_color, label_2 in runs:
            plt.plot(epochs, np.array(run_measurements[m_name])*(9 if "2" in m_name else 1), run_color+linetype, label=label_1 + label_2)

    for subplot, title in subplots:
        ax = plt.subplot(*subplot)
        plt.legend()
        plt.grid()
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.title(title)

    plt.tight_layout()
    plt.savefig("eps1_eps2.pdf")
    plt.show()



    # plt.plot(epochs, c1fc['SQI_eps1_avg'], 'ro-', label='$\epsilon^{avg}$, cifar10, 1FC')
    # plt.plot(epochs, c2fc['SQI_eps1_avg'], 'bo-', label='$\epsilon^{avg}$, cifar10, 2FC')
    # plt.plot(epochs, m2fc['SQI_eps1_avg'], 'ko-', label='$\epsilon^{avg}$, mnist, 2FC')
    # plt.plot(epochs, np.array(c1fc['SQI_eps2_avg'])*9, 'rx--', label=r'$(C-1)\tilde\epsilon^{avg}$, cifar10, 1FC')
    # plt.plot(epochs, np.array(c2fc['SQI_eps2_avg'])*9, 'bx--', label=r'$(C-1)\tilde\epsilon^{avg}$, cifar10, 2FC')
    # plt.plot(epochs, np.array(m2fc['SQI_eps2_avg'])*9, 'kx--', label=r'$(C-1)\tilde\epsilon^{avg}$, mnist, 2FC')
    # plt.legend()
    # plt.grid()
    # plt.yscale('log')
    # plt.xlabel('Epoch')
    # plt.title(r"Average $\epsilon_{i, c}$ and $(C-1)\tilde\epsilon_{i, c, c'}$")

    # # plt.show()

    # ax = plt.subplot(1, 2, 2)
    # plt.plot(epochs, c1fc['SQI_eps1_rel_std'], 'ro-', label=r'std$(\epsilon / \epsilon^{avg})$, cifar10, 1FC')
    # plt.plot(epochs, c2fc['SQI_eps1_rel_std'], 'bo-', label=r'std$(\epsilon / \epsilon^{avg})$, cifar10, 2FC')
    # plt.plot(epochs, m2fc['SQI_eps1_rel_std'], 'ko-', label=r'std$(\epsilon / \epsilon^{avg})$, mnist, 2FC')
    # plt.plot(epochs, np.array(c1fc['SQI_eps2_rel_std']), 'rx--', label=r'std$(\tilde\epsilon / \tilde\epsilon^{avg})$, cifar10, 1FC')
    # plt.plot(epochs, np.array(c2fc['SQI_eps2_rel_std']), 'bx--', label=r'std$(\tilde\epsilon / \tilde\epsilon^{avg})$, cifar10, 2FC')
    # plt.plot(epochs, np.array(m2fc['SQI_eps2_rel_std']), 'kx--', label=r'std$(\tilde\epsilon / \tilde\epsilon^{avg})$, mnist, 2FC')
    # plt.legend()
    # plt.grid()
    # plt.yscale('log')
    # plt.xlabel('Epoch')
    # plt.title(r'Empirical relative standard-deviation')


def plot_stl():
    # c1fc = get_measurements('cifar_1fc')
    c2fc, c2fc_stl = get_both('cifar_2fc')
    # m2fc, m2fc_stl = get_both('mnist_2fc')

    plt.figure(figsize=(16, 4))

    runs = [(c2fc_stl, 'bo', 'second-to-last layer'),
            (c2fc, 'rx-', 'last layer'),
            # (m2fc_stl, 'kx', 'mnist')
            ]
    subplots = [((1, 5, 1), "loss"),
                ((1, 5, 2), r'NC1: Tr$(\Sigma_w \Sigma_B^{-1})$'),
                ((1, 5, 3), r'NC2: Equiangle'),
                ((1, 5, 4), r'NC2: Equinorm'),
                # ((1, 5, 5), r'NC3: Self-duality'),
                ((1, 5, 5), r'NC4: NCC-mismatch'),
                ]
    measures = [('loss', '-', 0, ''),
                ('Sw_invSb', '-', 1, ''),
                ('cos_M', '-', 2, ''),
                ('norm_M_CoV', '-', 3, ''),
                # ('W_M_dist', '-', 4, ''),
                ('NCC_mismatch', '-', 4, ''),
                ]
    for m_name, linetype, subplot, label_1 in measures:
        ax = plt.subplot(*subplots[subplot][0])
        for run_measurements, run_color, label_2 in runs:
            plt.plot(epochs, run_measurements[m_name], run_color + linetype,
                     label=(label_1 + label_2 if m_name != 'loss' else None))

    for subplot, title in subplots:
        ax = plt.subplot(*subplot)
        plt.legend()
        plt.grid()
        # plt.yscale('log')
        plt.xlabel('Epoch')
        plt.title(title)

    plt.tight_layout()
    plt.savefig("stl_nc.pdf")
    plt.show()


if __name__ == "__main__":
    plot_stl()
    # plot_eps1_2()
