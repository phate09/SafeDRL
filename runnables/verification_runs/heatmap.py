import jsonpickle
import matplotlib.pyplot as plt
import numpy as np
import progressbar as pb
import ray
from verification_runs.cartpole_bab_load import generateCartpoleDomainExplorer

with open("../save/aggregated_safe_domains.json", 'r') as f:
    frozen_safe = jsonpickle.decode(f.read())
with open("../save/aggregated_unsafe_domains.json", 'r') as f:
    frozen_unsafe = jsonpickle.decode(f.read())
# with open("../save/ignore_domains.json", 'r') as f:
#     frozen_ignore = jsonpickle.decode(f.read())
frozen_safe = np.stack(frozen_safe)  # .take(range(10), axis=0)
frozen_unsafe = np.stack(frozen_unsafe)  # .take(range(10), axis=0)
explorer, verification_model = generateCartpoleDomainExplorer()


# frozen_unsafe = np.stack(frozen_unsafe)  # .take(range(10), axis=0)
# frozen_ignore = np.stack(frozen_ignore)  # .take(range(10), axis=0)
# @ray.remote
def contains(first, second, safe_domains, unsafe_domains):
    domains = np.concatenate((unsafe_domains, safe_domains))
    for i, domain in enumerate(domains):
        if domain[0, 0] <= first <= domain[0, 1] and domain[2, 0] <= second <= domain[2, 1]:
            if i < len(safe_domains):
                return -1
            else:
                return 1
    return 0


def heatmap(original_domain, safe_domains, unsafe_domains, points_x=50, points_y=50, parallelise=True):
    if parallelise:
        ray.init()
    linx = np.linspace(0, 1, num=points_x)  # since the domains are normalised we use a linear space between 0 and 1
    liny = np.linspace(0, 1, num=points_y)
    values = np.zeros((points_x, points_y))
    returns = []
    widget = ['building heatmap: ', pb.Percentage(), ' ',
              pb.Bar(), ' ', pb.ETA()]
    timer = pb.ProgressBar(widgets=widget, maxval=points_x * points_y).start()
    steps = 0
    for i in range(points_x):
        for j in range(points_y):
            if parallelise:
                if len(returns) > 9:
                    id, i, j = returns.pop(0)
                    values[i, j] = ray.get(id)
                returns.append((contains.remote(linx[i], liny[j], safe_domains, unsafe_domains), i, j))
                steps += 1
                timer.update(steps)
            else:
                values[i, j] = contains(linx[i], liny[j], safe_domains, unsafe_domains)
                steps += 1
                timer.update(steps)
    for _ in range(len(returns)):
        id, i, j = returns.pop(0)
        values[i, j] = ray.get(id)
    timer.finish()
    fig, ax = plt.subplots()
    im = ax.imshow(values)
    n_ticks_x = 8
    n_ticks_y = 8

    ax.set_xticks(np.linspace(0, points_x, num=n_ticks_x))
    ax.set_yticks(np.linspace(0, points_y, num=n_ticks_y))
    ax.set_xticklabels(np.linspace(original_domain[0, 0], original_domain[0, 1], num=n_ticks_x))
    ax.set_yticklabels(np.linspace(original_domain[2, 0], original_domain[2, 1], num=n_ticks_y))
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.colorbar(im)
    fig.tight_layout()
    plt.show()


heatmap(explorer.initial_domain.cpu().numpy(), frozen_safe, frozen_unsafe, points_x=100, points_y=100, parallelise=False)
