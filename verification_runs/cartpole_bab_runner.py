import jsonpickle

from verification_runs.cartpole_bab_load import generateCartpoleDomainExplorer

explorer, verification_model = generateCartpoleDomainExplorer()

explorer.explore(verification_model)
frozen_safe = jsonpickle.encode([i.cpu().numpy() for i in explorer.safe_domains])
frozen_unsafe = jsonpickle.encode([i.cpu().numpy() for i in explorer.unsafe_domains])
frozen_ignore = jsonpickle.encode([i.cpu().numpy() for i in explorer.ignore_domains])

with open("../runs/safe_domains.json", 'w+') as f:
    f.write(frozen_safe)
with open("../runs/unsafe_domains.json", 'w+') as f:
    f.write(frozen_unsafe)
with open("../runs/ignore_domains.json", 'w+') as f:
    f.write(frozen_ignore)
