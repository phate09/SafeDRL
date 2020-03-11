from prism.state_storage import get_storage
import ray


@ray.remote
def count(i: int):
    with get_storage() as c:
        return c.store([(i, i + 1)])


ray.init()
ids = [count.remote(i) for i in range(10)]
results = ray.get(ids)
print(results)  # c2 = get_storage()
# print(c.store("Edoardo"))
# print(c2.store("Edoardo2"))
# print(c.store("Edoardo"))
# print(c2.store("Edoardo2"))
