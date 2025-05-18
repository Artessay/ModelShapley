from generation import generate
from evaluation import evaluate
from deactivate import deactivate_neurons
from utils import get_args

def main():
    args = get_args()
    if args.mode not in ['vanilla'] and args.train == False:
        deactivate_neurons(args)
    generate(args)
    evaluate(args)

if __name__ == "__main__":
    main()