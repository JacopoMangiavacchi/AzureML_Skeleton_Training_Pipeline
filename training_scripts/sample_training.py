
"""
This module implement the sample model training main script.
"""
import os
import pandas as pd

from azureml.core import Run

from sklearn.externals import joblib

LOCAL_MODEL_FILE = "samle.pkl"

def train(data_dir, input_csv_file, sparcity_threshold):
    """
    Main train script for the Sample Model Training
    """
    # Read the Data From File
    # features_df = pd.read_csv(os.path.join(data_dir, input_csv_file))

    # train model
    model = {}

    # locally save model
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=model, filename="outputs/" + LOCAL_MODEL_FILE)
    print("Model saved")


if __name__ == "__main__":
    RUN = Run.get_context()
    RUN.log("Training step", 1)

    # Get Parameters
    PARSER = argparse.ArgumentParser("training")
    PARSER.add_argument('--input', type=str, \
        dest='data_dir', help='data storage reference in AML', required=True)
    PARSER.add_argument('--input_csv_file', \
        type=str, help='sales csv file in the input folder', required=True)
    PARSER.add_argument('--model_name', \
        type=str, help='model name', required=True)

    ARGS = PARSER.parse_args()

    print("In basemodel_training.py")

    # Train / Test / Upload Model
    train(ARGS.data_dir, ARGS.input_csv_file)

    RUN.log("Training step", 2)
    print("Model trained")

    # Upload model
    print(RUN.get_file_names())
    RUN.upload_file(LOCAL_MODEL_FILE, "outputs/" + LOCAL_MODEL_FILE)
    MODEL = RUN.register_model(model_name=ARGS.model_name, model_path=LOCAL_MODEL_FILE)
    print(MODEL.name, MODEL.id, MODEL.version, sep='\t')
    RUN.log("Training step", 3)
    print("Model registered")

    RUN.complete()
