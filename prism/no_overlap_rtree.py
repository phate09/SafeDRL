import Pyro5.api
import ray

from symbolic.unroll_methods import NoOverlapRtree


if __name__ == '__main__':
    local_mode = False
    if not ray.is_initialized():
        ray.init(local_mode=local_mode, include_webui=True, log_to_driver=False)
    n_workers = int(ray.cluster_resources()["CPU"]) if not local_mode else 1
    Pyro5.api.config.SERIALIZER = "marshal"
    Pyro5.api.config.SERVERTYPE = "multiplex"
    Pyro5.api.Daemon.serveSimple({NoOverlapRtree: "prism.rtreetemp"}, ns=True)
