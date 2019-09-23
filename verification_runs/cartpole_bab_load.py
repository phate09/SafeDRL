import jsonpickle
import numpy as np
from verification_runs.aggregate_abstract_domain import aggregate


def main():
    with open("../runs/safe_domains.json", 'r') as f:
        frozen_safe = jsonpickle.decode(f.read())
    with open("../runs/unsafe_domains.json", 'r') as f:
        frozen_unsafe=jsonpickle.decode(f.read())
    with open("../runs/ignore_domains.json", 'r') as f:
        frozen_ignore=jsonpickle.decode(f.read())
    frozen_safe = np.stack(frozen_safe)  # .take(range(10), axis=0)
    frozen_unsafe = np.stack(frozen_unsafe)  # .take(range(10), axis=0)
    frozen_ignore = np.stack(frozen_ignore)  # .take(range(10), axis=0)
    aggregated_safe = aggregate(frozen_safe)
    print(f"safe {{{frozen_safe.shape} --> {aggregated_safe.shape}}}")
    aggregated_unsafe = aggregate(frozen_unsafe)
    print(f"unsafe {{{frozen_unsafe.shape} --> {aggregated_unsafe.shape}}}")
    aggregated_ignore = aggregate(frozen_ignore)
    print(f"ignore {{{frozen_ignore.shape} --> {aggregated_ignore.shape}}}")




if __name__ == '__main__':
    main()
