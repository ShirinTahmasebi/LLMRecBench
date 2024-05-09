def singleton(cls):
    instance = None
    def get_instance(*args, **kwargs):
        nonlocal instance
        if instance is None:
            instance = cls(*args, **kwargs)
        return instance
    cls.get_instance = staticmethod(get_instance)
    return cls