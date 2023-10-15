


def build_object_from_class_name(cls_str, parent_cls, strict=False, *args, **kwargs):
    import importlib

    pkg = ".".join(cls_str.split(".")[:-1])
    cls_name = cls_str.split(".")[-1]
    cls_type = getattr(importlib.import_module(pkg), cls_name)
    if parent_cls is not None:
        assert issubclass(cls_type, parent_cls), f'| {cls_type} is not subclass of {parent_cls}.'
    if strict:
        return cls_type(*args, **kwargs)
    return cls_type(*args, **filter_kwargs(kwargs, cls_type))

def build_object_from_class_name_opt(cls_str, parent_cls,  *args, **kwargs):
    import importlib

    pkg = ".".join(cls_str.split(".")[:-1])
    cls_name = cls_str.split(".")[-1]
    cls_type = getattr(importlib.import_module(pkg), cls_name)
    if parent_cls is not None:
        assert issubclass(cls_type, parent_cls), f'| {cls_type} is not subclass of {parent_cls}.'

    return cls_type(*args, **filter_kwargs(kwargs, cls_type))



def build_lr_scheduler_from_config(optimizer, scheduler_args):
    try:
        # PyTorch 2.0+
        from torch.optim.lr_scheduler import LRScheduler as LRScheduler
    except ImportError:
        # PyTorch 1.X
        from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

    def helper(params):
        if isinstance(params, list):
            return [helper(s) for s in params]
        elif isinstance(params, dict):
            resolved = {k: helper(v) for k, v in params.items()}
            if 'cls' in resolved:
                if (
                    resolved["cls"] == "torch.optim.lr_scheduler.ChainedScheduler"
                    and scheduler_args["scheduler_cls"] == "torch.optim.lr_scheduler.SequentialLR"
                ):
                    raise ValueError(f"ChainedScheduler cannot be part of a SequentialLR.")
                resolved['optimizer'] = optimizer
                obj = build_object_from_class_name_opt(
                    resolved['cls'],
                    LRScheduler,
                    **resolved
                )
                return obj
            return resolved
        else:
            return params

    resolved = helper(scheduler_args)
    resolved['optimizer'] = optimizer
    return build_object_from_class_name_opt(
        scheduler_args['scheduler_cls'],
        LRScheduler,
        **resolved
    )



def filter_kwargs(dict_to_filter, kwarg_obj):
    import inspect

    sig = inspect.signature(kwarg_obj)
    filter_keys = [param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
    filtered_dict = {filter_key: dict_to_filter[filter_key] for filter_key in filter_keys if
                     filter_key in dict_to_filter}
    return filtered_dict