from models.base import PoissonBase, load_data

class PoissonDecay(PoissonBase):
    """Modelo Poisson con decay temporal (pesos exponenciales)."""

    def fit(self, df, decay=0.004):
        return super().fit(df, decay=decay)
