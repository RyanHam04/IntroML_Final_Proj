from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve


# This code is almost directly extracted from:
# https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_display_object_visualization.html
def plot_pr_curve(y_test, y_proba):
    prec, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
