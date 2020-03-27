import Pyro5.api

from prism.shared_dictionary import SharedDict
from prism.shared_rtree import SharedRtree
from prism.shared_rtree_temp import SharedRtree_temp
from prism.state_storage import StateStorage

if __name__ == '__main__':
    Pyro5.api.Daemon.serveSimple({SharedRtree: "prism.rtree",
                                  SharedRtree_temp: "prism.rtreetemp",
                                  SharedDict: "prism.shareddict",
                                  StateStorage: "prism.statestorage"}, ns=True)
