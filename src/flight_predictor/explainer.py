import shap
import matplotlib.pyplot as plt
from pathlib import Path


class Explainer:

    def __init__(self, reports_dir: str = "reports/figures"):
        self.reports_dir = Path(reports_dir)
        self._explainer = None
        self._shap_values = None

    def fit(self, model, X_sample) -> "Explainer":
        self._explainer = shap.TreeExplainer(model)
        self._shap_values = self._explainer(X_sample)
        return self

    def summary_bar(self):
        self._check_fitted()
        shap.summary_plot(
            self._shap_values,
            plot_type="bar",
            show=False
        )
        self._save("shap_summary_bar.png")

    def summary_dot(self):
        self._check_fitted()
        shap.summary_plot(
            self._shap_values,
            show=False
        )
        self._save("shap_summary_dot.png")

    def waterfall(self, sample_index: int = 0):
        self._check_fitted()
        shap.plots.waterfall(
            self._shap_values[sample_index],
            show=False
        )
        self._save("shap_waterfall.png")

    def _save(self, filename: str):
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.reports_dir / filename, bbox_inches="tight")
        plt.close()

    def _check_fitted(self):
        if self._explainer is None or self._shap_values is None:
            raise RuntimeError("Call fit() before generating plots.")
