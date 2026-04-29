#!/usr/bin/env python3

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


def main():

    try:
        # Import seeding module
        from scripts.seed_demo import main as seed_main

        seed_main()

        
    except Exception as e:
        sys.exit(1)


if __name__ == "__main__":
    main()
