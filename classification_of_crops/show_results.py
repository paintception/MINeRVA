import pickle
import argparse


def arguments():
    parser = argparse.ArgumentParser(description='arguments')

    parser.add_argument('--f', type=str, help='name', default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()
    file = args.f

    with open(file, "rb") as openfile:
        print(pickle.load(openfile))
