import argparse
from train import train_model
from evaluate import evaluate_model
from plot_features import plot
from predict import predict_claims

def main():
    ag = argparse.ArgumentParser()
    ag.add_argument(
        "--train", action="store_true", help="Retrain the model before running"
    )
    clargs = ag.parse_args()
    
    if clargs.train:
        train_model()

    evaluate_model()
    plot()
    predict_claims()
    
if __name__ == "__main__":
    main()
