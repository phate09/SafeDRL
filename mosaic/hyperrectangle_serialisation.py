import ray

from mosaic.hyperrectangle import HyperRectangle_action, HyperRectangle


def ray_serializer_hyper_action(obj):
    return obj.to_numpy()


def ray_serializer_hyper(obj):
    return obj.to_numpy()


def ray_deserializer_hyper_action(value):
    return HyperRectangle_action.from_numpy(value)


def ray_deserializer_hyper(value):
    return HyperRectangle.from_numpy(value)


def register_serialisers():
    ray.register_custom_serializer(HyperRectangle_action, serializer=ray_serializer_hyper_action, deserializer=ray_deserializer_hyper_action)
    ray.register_custom_serializer(HyperRectangle, serializer=ray_serializer_hyper, deserializer=ray_deserializer_hyper)
