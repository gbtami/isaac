"""Module entrypoint for `python -m isaac.client`."""

from __future__ import annotations

import asyncio
import sys

from isaac.client.client import main


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main(sys.argv)))
    except KeyboardInterrupt:
        raise SystemExit(130)
