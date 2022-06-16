import argparse

from customGameWithMultipleAgents import simulate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, required=True)

    # colab env arguments:
    parser.add_argument('--firstHeuristic', type=str, required=False)
    parser.add_argument('--secondHeuristic', type=str, required=False)

    args = parser.parse_args()

    if args.env == "colab":
        first_heuristic = args.firstHeuristic
        second_heuristic = args.secondHeuristic
        simulate(first_heuristic, second_heuristic)
    else:
        print("ENVIRONMENT NOT ABAILABLE")