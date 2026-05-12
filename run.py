

from sklearn.model_selection import train_test_split
from flight_predictor.data_loader import DataLoader
from flight_predictor.preprocessor import Preprocessor
from flight_predictor.feature_engineer import FeatureEngineer
from flight_predictor.target_encoder import TargetEncoder
from flight_predictor.trainer import ModelTrainer
from flight_predictor.evaluator import ModelEvaluator
from flight_predictor.explainer import Explainer


def split(df):
    X = df.drop(columns=["price"])
    y = df["price"]


    # First split off test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Then split remaining into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.2,
        random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":



        print("Loading data...")
        loader = DataLoader()
        df_raw = loader.load()

        print("Preprocessing...")
        preprocessor = Preprocessor()
        df_clean = preprocessor.fit_transform(df_raw)

        print("Feature engineering...")
        engineer = FeatureEngineer()
        df_featured = engineer.fit_transform(df_clean)

        print("Splitting...")
        X_train, X_val, X_test, y_train, y_val, y_test = split(df_featured)

        print("Encoding...")
        encoder = TargetEncoder()
        X_train = encoder.fit_transform(X_train, y_train)
        X_val = encoder.transform(X_val)
        X_test = encoder.transform(X_test)

        print("Training...")
        trainer = ModelTrainer()
        trainer.train(X_train, y_train, X_val, y_val)
        trainer.save()





        print("Evaluating...")
        evaluator = ModelEvaluator()
        val_metrics = evaluator.evaluate(trainer.model, X_val, y_val)
        test_metrics = evaluator.evaluate(trainer.model, X_test, y_test)
        print(evaluator.report(trainer.model, {
            "val":  (X_val,  y_val),
            "test": (X_test, y_test)
        }))






        print("Done.")

    
