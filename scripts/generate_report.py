#!/usr/bin/env python3
import sys
from cli import main

if __name__ == "__main__":
    main(["--stage", "report", *sys.argv[1:]])
