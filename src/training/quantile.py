import numpy as np
import lightgbm as lgb

class BaselineHistorique:
    def __init__(self, couverture:float) -> None:
        self.couverture = couverture

        self.q_low = (1 - self.couverture)/2
        self.q_up = 1 - (1 - self.couverture)/2

        self.intervalle_fixe = None

    def fit(self, y_train: list) -> "BaselineHistorique":
        y_train = np.array(y_train)

        self.intervalle_fixe = np.quantile(y_train, [self.q_low, self.q_up])
    
        return self

    def predict(self, y_test: list) -> np.ndarray:
        if self.intervalle_fixe is None:
            raise ValueError("Le modèle doit d'abord être entraîné avec la méthode .fit(y_train)")

        n_echantillons = len(y_test)
        
        preds = np.tile(self.intervalle_fixe, (n_echantillons, 1))

        return preds


class BaselineLightGBM:
    def __init__(self, couverture: float, **kwargs) -> None:
        self.couverture = couverture

        self.q_low = (1 - self.couverture) / 2
        self.q_up = 1 - self.q_low

        self.model_low = lgb.LGBMRegressor(
            objective="quantile", alpha=self.q_low, **kwargs
        )
        self.model_up = lgb.LGBMRegressor(
            objective="quantile", alpha=self.q_up, **kwargs
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BaselineLightGBM":
        self.model_low.fit(X_train, y_train)
        self.model_up.fit(X_train, y_train)

        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        pred_low = self.model_low.predict(X_test)
        pred_up = self.model_up.predict(X_test)

        pred_low, pred_up = np.minimum(pred_low, pred_up), np.maximum(pred_low, pred_up)

        preds = np.stack([pred_low, pred_up], axis=1)

        return preds
    

if __name__ == "__main__":
    # Test avec des donnees factices
    np.random.seed(42)
    X_train = np.random.rand(1000, 10)
    y_train = X_train.sum(axis=1) + np.random.normal(0, 0.1, 1000)
    X_test = np.random.rand(500, 10)
    y_test = X_test.sum(axis=1) + np.random.normal(0, 0.1, 500)

    modeles = {
        "BaselineHistorique": BaselineHistorique(couverture=0.9).fit(y_train),
        "BaselineLightGBM": BaselineLightGBM(couverture=0.9, verbose=-1).fit(X_train, y_train),
    }

    for nom, modele in modeles.items():
        preds = modele.predict(X_test)
        lower, upper = preds[:, 0], preds[:, 1]

        print(f"--- {nom} ---")
        print("preds.shape:", preds.shape)
        print("couverture:", np.mean((y_test >= lower) & (y_test <= upper)))
        print("largeur moyenne:", np.mean(upper - lower))
