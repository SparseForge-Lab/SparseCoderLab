"""Lazy public model import avoids a router/layers package initialization cycle."""
__all__ = ['LanguageModel']

def __getattr__(name):
    if name == 'LanguageModel':
        from .transformer import LanguageModel
        return LanguageModel
    raise AttributeError(name)
