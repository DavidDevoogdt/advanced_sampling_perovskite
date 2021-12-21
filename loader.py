import src
from src.bg import bg
import pickle


def main():
    src.config = pickle.load(open('config.pickle', 'rb'))
    if src.config.bg:
        bg()


if __name__ == "__main__":
    main()
