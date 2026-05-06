# Torna a pasta preprocessing um módulo Python importável
from preprocessing.resize import apply_resize
from preprocessing.grayscale import apply_grayscale

__all__ = ["apply_resize", "apply_grayscale"]
