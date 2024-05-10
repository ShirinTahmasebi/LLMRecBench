def singleton(cls):
    instance = None
    def get_instance(*args, **kwargs):
        nonlocal instance
        if instance is None:
            instance = cls(*args, **kwargs)
        return instance
    cls.get_instance = staticmethod(get_instance)
    return cls


# For creating constant variables (public static final variables)
# https://stackoverflow.com/a/2688086
def constant(f):
    def fset(self, value):
        raise TypeError

    def fget(self):
        return f()
    return property(fget, fset)