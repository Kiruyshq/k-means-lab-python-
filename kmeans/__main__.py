"""Allow running the package as `python -m kmeans`."""

import sys

from kmeans.cli import main

sys.exit(main())
