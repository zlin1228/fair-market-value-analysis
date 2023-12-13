from resource import getrusage, getpagesize, RUSAGE_SELF
import time
from typing import Union

_PAGESIZE = getpagesize()

def _get_current_memory() -> int:
    with open('/proc/self/statm', mode='rb') as f:
        return int(f.readline().split()[1]) * _PAGESIZE

class _Profile:
    def __init__(self, msg: str, prefix: str='', ndigits: int=3) -> None:
        self._msg: str = msg
        self._prefix: str = f"[{prefix}] " if prefix else ""
        self._ndigits: int = ndigits

    def __enter__(self) -> '_Profile':
        self._start: float = time.time()
        print(f'{self._prefix}{self._msg}...', flush=True)
        return self

    def __exit__(self, *args) -> None:
        print(f'{self._prefix}{self._msg}{" " if self._msg else ""}time: {round(time.time() - self._start, self._ndigits)}s, current memory: {round(_get_current_memory() / 1073741824, self._ndigits)}GB, max memory: {round(getrusage(RUSAGE_SELF).ru_maxrss / 1048576, self._ndigits)}GB', flush=True)

    def elapsed_time(self) -> float:
        return time.time() - self._start

    def print(self, msg) -> None:
        print(f'{self._prefix}{msg}')

def create_profiler(prefix: str='', ndigits: int=3):
    _ndigits: int = ndigits
    def profile(msg: str='', ndigits: Union[int,None]=None) -> _Profile:
        return _Profile(msg, prefix, _ndigits if ndigits is None else ndigits)
    return profile
