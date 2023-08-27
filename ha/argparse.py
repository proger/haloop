from argparse import *

class Formatter(
    ArgumentDefaultsHelpFormatter,
    MetavarTypeHelpFormatter,
    RawDescriptionHelpFormatter,
):
    pass
