from sklearn.ensemble import HistGradientBoostingClassifier

def get_model():
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=8,
        max_iter=400,
        min_samples_leaf=20
    )
