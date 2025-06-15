from argparse import *

class Formatter(
    ArgumentDefaultsHelpFormatter,
    MetavarTypeHelpFormatter,
    RawDescriptionHelpFormatter,
):
    pass


def int_or_float(value):
    try:
        # Try parsing as integer
        int_val = int(value)
        if '.' in value or 'e' in value.lower():
            # Contains float-specific syntax; parse as float
            return float(value)
        return int_val
    except ValueError:
        # Fallback to float
        try:
            return float(value)
        except ValueError:
            raise ArgumentTypeError(f"{value} is not a valid int or float")
