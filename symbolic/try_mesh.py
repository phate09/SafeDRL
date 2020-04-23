import random
import plotly.graph_objects as go
import ray

from mosaic.utils import show_plot3d
from symbolic.mesh_cube import get_mesh
from symbolic.unroll_methods import merge_supremum3


def create_random_domain(dimensions):
    domain = []
    for d in range(dimensions):
        while True:
            n1 = random.randint(0, 10)
            n2 = random.randint(0, 10)
            if n1 != n2:
                break
        domain.append((min(n1, n2), max(n1, n2)))
    return tuple(domain)


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    dimensions = 3
    domains = []
    for i in range(10):
        domains.append(create_random_domain(dimensions))
    if not ray.is_initialized():
        ray.init(local_mode=True)
    merged_domains = merge_supremum3([(x, True) for x in domains], 8)
    show_plot3d(domains, merged_domains)
