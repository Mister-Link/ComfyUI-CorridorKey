from .loader import NODE_CLASS_MAPPINGS as _loader_classes, NODE_DISPLAY_NAME_MAPPINGS as _loader_names
from .keyer import NODE_CLASS_MAPPINGS as _keyer_classes, NODE_DISPLAY_NAME_MAPPINGS as _keyer_names

NODE_CLASS_MAPPINGS = {**_loader_classes, **_keyer_classes}
NODE_DISPLAY_NAME_MAPPINGS = {**_loader_names, **_keyer_names}
