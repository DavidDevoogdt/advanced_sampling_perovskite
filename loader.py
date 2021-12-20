import src
from src.bg import bg
import pickle


def main():
    if src.config.do_bg:
        bg()


if __name__ == "__main__":
    src.config = pickle.load(open('config.pickle', 'rb'))

    main()
