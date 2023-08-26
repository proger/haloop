from functools import wraps, reduce
import time
import shlex
import subprocess
from pathlib import Path
from contextlib import ExitStack


@wraps(subprocess.run)
def run(cmd, *args, output_filename: Path | None = None, **kwargs):
    with ExitStack() as stack:
        if output_filename:
            kwargs['stdout'] = stack.enter_context(open(output_filename, 'w'))
            kwargs['stderr'] = subprocess.STDOUT

        if isinstance(cmd, str):
            cmd = [cmd]
            kwargs['shell'] = True

        if output_filename:
            print(shlex.join(cmd), '>', output_filename)
        else:
            print(shlex.join(cmd))
        x = cmd[0]
        t0 = time.time()
        if not 'check' in kwargs:
            kwargs['check'] = True
        try:
            ret = subprocess.run(cmd, *args, **kwargs)
            return ret
        finally:
            t = time.time() - t0
            print('#', x, 'took', t)


def sh(x, *args, **kwargs):
    dash_dash = [[f"--{kw.replace('_', '-')}", str(kwargs[kw])] for kw in kwargs]
    return run([x] + reduce(list.__add__, dash_dash, [])  + [str(arg) for arg in args])
