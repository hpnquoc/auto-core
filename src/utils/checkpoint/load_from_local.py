from .loader_manager import CheckpointLoader
import torch
import os.path as osp
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Union

@CheckpointLoader.register_scheme(prefixes='')
def load_from_local(
    filename: str,
    map_location: Union[str, Callable, None] = None,
) -> Union[dict, OrderedDict]:
    """load checkpoint by local file path.

    Args:
        filename (str): local checkpoint file path
        map_location (str, optional): Same as :func:`torch.load`.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f'{filename} can not be found.')
    checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint
