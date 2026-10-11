"""Run a generated encoder recipe using the YAML launcher's local runtime setup.

Invoked by experiments/generated/encoder/*.local.sh. Accepts trainer arguments plus
--dry-run; validation and preview need no torch import. To select another backend or
override YAML settings with --set, use scripts/launch.py instead.
"""

from __future__ import annotations

import argparse
import os
import shlex

from scripts.launch import REPO_ROOT, encoder_local_command


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--dry-run", action="store_true")
    options, arguments = parser.parse_known_args(argv)
    command, env = encoder_local_command(arguments)
    if options.dry_run:
        print(shlex.join(command))
        return
    print(f"Running locally: {shlex.join(command)}", flush=True)
    os.chdir(REPO_ROOT)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
