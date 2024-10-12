"""Decorators for creating data abstractions."""

GETTERS = {}
SETTERS = {}


def gets(component):
    """Decorate function as a getter for a component."""

    def decorator(func):
        GETTERS[component] = func
        return func

    return decorator


def sets(component):
    """Decorate a function as a setter for a component."""

    def decorator(func):
        SETTERS[component] = func
        return func

    return decorator
