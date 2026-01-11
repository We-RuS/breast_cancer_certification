from matplotlib import pyplot as plt
from src.breast_cancer_classification.evaluation.visualizer import ModelVisualizer
from src.breast_cancer_classification.models.trainer import ModelTrainer
from src.breast_cancer_classification.data import loader
from src.breast_cancer_classification.models.classifier import get_all_classifiers
from src.breast_cancer_classification.evaluation.tables import save_models_comparison_table


def run():
    X, y, metadata = loader.load_raw_data()
    md = ModelTrainer()
    X_train, X_test, y_train, y_test = md.prepare_data(X, y)
    classifiers_names = list(get_all_classifiers().keys())
    results = md.train_all_models(X_train, y_train, classifiers_names)
    save_models_comparison_table(md)
    visualizer = ModelVisualizer()

    for model_name, i_results in results.items():
        model = i_results["Grid search"].best_estimator_
        y_pred = model.predict(X_test)

        visualizer.plot_confusion_matrix(y_test, y_pred)
        plt.savefig(f"reports/figures/confusion_matrix/{model_name}.png")
        visualizer.plot_roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        plt.savefig(f"reports/figures/roc_curve/{model_name}.png")
        visualizer.plot_learning_curves(model, X_test, y_test)
        plt.savefig(f"reports/figures/learning_curve/{model_name}.png")


if __name__ == "__main__":
    run()
