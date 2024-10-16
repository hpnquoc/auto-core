import torch

from .multi_step_lr import MultiStepLR_Restart
from .cosine_annealing_lr import CosineAnnealingLR_Restart


if __name__ == "__main__":
    optimizer = torch.optim.Adam(
        [torch.zeros(3, 64, 3, 3)], lr=2e-4, weight_decay=0, betas=(0.9, 0.99)
    )
    ##############################
    # MultiStepLR_Restart
    ##############################
    ## Original
    lr_steps = [200000, 400000, 600000, 800000]
    restarts = None
    restart_weights = None

    ## two
    lr_steps = [
        100000,
        200000,
        300000,
        400000,
        490000,
        600000,
        700000,
        800000,
        900000,
        990000,
    ]
    restarts = [500000]
    restart_weights = [1]

    ## four
    lr_steps = [
        50000,
        100000,
        150000,
        200000,
        240000,
        300000,
        350000,
        400000,
        450000,
        490000,
        550000,
        600000,
        650000,
        700000,
        740000,
        800000,
        850000,
        900000,
        950000,
        990000,
    ]
    restarts = [250000, 500000, 750000]
    restart_weights = [1, 1, 1]

    scheduler = MultiStepLR_Restart(
        optimizer, lr_steps, restarts, restart_weights, gamma=0.5, clear_state=False
    )

    ##############################
    # Cosine Annealing Restart
    ##############################
    ## two
    T_period = [500000, 500000]
    restarts = [500000]
    restart_weights = [1]

    ## four
    T_period = [250000, 250000, 250000, 250000]
    restarts = [250000, 500000, 750000]
    restart_weights = [1, 1, 1]

    scheduler = CosineAnnealingLR_Restart(
        optimizer, T_period, eta_min=1e-7, restarts=restarts, weights=restart_weights
    )

    ##############################
    # Draw figure
    ##############################
    N_iter = 1000000
    lr_l = list(range(N_iter))
    for i in range(N_iter):
        optimizer.step()
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        lr_l[i] = current_lr

    import matplotlib as mpl
    import matplotlib.ticker as mtick
    from matplotlib import pyplot as plt

    mpl.style.use("default")
    import seaborn

    seaborn.set_theme(style="whitegrid")
    seaborn.set_context("paper")

    plt.figure(1)
    plt.subplot(111)
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    plt.title("Title", fontsize=16, color="k")
    plt.plot(list(range(N_iter)), lr_l, linewidth=1.5, label="learning rate scheme")
    legend = plt.legend(loc="upper right", shadow=False)
    ax = plt.gca()
    labels = ax.get_xticks().tolist()
    for k, v in enumerate(labels):
        labels[k] = str(int(v / 1000)) + "K"
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))

    ax.set_ylabel("Learning rate")
    ax.set_xlabel("Iteration")
    fig = plt.gcf()
    plt.show()