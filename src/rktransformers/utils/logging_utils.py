# Copyright 2025 Emmanuel Cortes. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for logging"""

import contextlib
import os
import sys


@contextlib.contextmanager
def suppress_output():
    """
    Context manager to suppress stdout and stderr from RKNN/RKNNLite C-level logs.

    This context manager redirects file descriptors 1 (stdout) and 2 (stderr)
    to /dev/null, suppressing output at the OS level. This is necessary for
    silencing C-level output from libraries like RKNN/RKNNLite that bypass
    Python's stdout/stderr redirection.
    """
    null_fds = []
    save_fds = []

    try:
        # Flush Python buffers to ensure order (if they still exist)
        if sys.stdout is not None:
            sys.stdout.flush()
        if sys.stderr is not None:
            sys.stderr.flush()

        # Open a pair of null files
        null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        save_fds = [os.dup(1), os.dup(2)]

        # Assign the null pointers to stdout and stderr.
        os.dup2(null_fds[0], 1)
        os.dup2(null_fds[1], 2)

        yield
    except Exception:
        # Silently handle any errors during setup or body execution
        pass
    finally:
        with contextlib.suppress(Exception):
            # Flush again to be safe (if stdout/stderr still exist)
            if sys.stdout is not None:
                sys.stdout.flush()
            if sys.stderr is not None:
                sys.stderr.flush()

            # Re-assign the real stdout/stderr back to (1) and (2)
            if save_fds:
                os.dup2(save_fds[0], 1)
                os.dup2(save_fds[1], 2)

            # Close the null files and saved fds
            for fd in null_fds + save_fds:
                os.close(fd)
