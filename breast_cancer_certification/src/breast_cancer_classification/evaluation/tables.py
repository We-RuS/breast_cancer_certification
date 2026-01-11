import pandas as pd
from tabulate import tabulate

from src.breast_cancer_classification.models.trainer import ModelTrainer


def save_models_comparison_table(trainer: ModelTrainer, filename: str = "results_table") -> pd.DataFrame:
    """
    Saves model comparison table.

    :param: trainer: ModelTrainer
        Trainer object.
    :param: filename: str
        Base filename for table.

    :return: pd.DataFrame
        Table with models comparison results.
    """
    comparison_df = trainer.get_comparison_table()

    csv_path = f"reports/results/{filename}.csv"
    comparison_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ Таблица сравнения сохранена: {csv_path}")

    md_path = f"reports/results/{filename}.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Model Comparison Report\n\n")
        f.write(tabulate(comparison_df, headers='keys', tablefmt='github', showindex=False))
        f.write(f"\n\n## Summary\n")

        best_model = comparison_df.iloc[0]
        f.write(f"- **Best Model**: {best_model['Model']}\n")
        f.write(f"- **Best CV Score**: {best_model['CV Accuracy']:.4f}\n")
        if 'Test Score' in comparison_df.columns:
            f.write(f"- **Test Score**: {best_model['Test Score']:.4f}\n")

    print(f"✅ Markdown отчет сохранен: {md_path}")

    return comparison_df


def create_summary_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Creates summary table with statistic."""
    summary_data = []

    for metric in ['CV Score', 'Test Score']:
        if metric in comparison_df.columns:
            summary_data.append({
                'Metric': metric,
                'Mean': comparison_df[metric].mean(),
                'Std': comparison_df[metric].std(),
                'Min': comparison_df[metric].min(),
                'Max': comparison_df[metric].max(),
                'Best Model': comparison_df.loc[comparison_df[metric].idxmax(), 'Model']
            })

    summary_df = pd.DataFrame(summary_data)

    for col in ['Mean', 'Std', 'Min', 'Max']:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].apply(lambda x: f"{x:.4f}")

    return summary_df
