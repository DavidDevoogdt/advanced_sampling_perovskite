from src.bg import bg
import config
import sys
import functools

print = functools.partial(print, flush=config.debug)


def main(debug=False):

    if config.do_bg:
        bg()


if __name__ == "__main__":
    if sys.argv == 2:
        debug = sys.argv[1] == "True"
    else:
        debug = config.debug

    main(debug)
