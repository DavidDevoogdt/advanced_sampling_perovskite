from src.bg import bg
import config


def main():
    if config.do_bg:
        bg()


if __name__ == "__main__":
    main()
