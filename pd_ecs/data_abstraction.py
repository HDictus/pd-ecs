GETTERS = {}
SETTERS = {}


def gets(component):

    def decorator(func):
        GETTERS[component] = func
        return func

    return decorator


def sets(component):

    def decorator(func):
        SETTERS[component] = func
        return func

    return decorator
