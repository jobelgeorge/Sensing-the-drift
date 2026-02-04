import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

class SideSlipWeighter:
    """
    Maps a sideslip angle value to:
        - band: 'many', 'medium', or 'few'
        - weight: numeric importance used in loss

    Uses sideSlip_distribution.csv provided in the dataset.
    """

    def __init__(self, csv_path, band_weights, round_to=1, use_lds=False, lds_sigma=1.0):
        """
        Parameters
        ----------
        csv_path : str
            Path to sideSlip_distribution.csv

        band_weights : dict
            Example:
                {
                    "many": 1.0,
                    "medium": 3.0,
                    "few": 8.0
                }

        round_to : int
            Decimal places to round sideslip before lookup.
            Must match CSV resolution (0.1 → round_to=1).
        """
        self.round_to = round_to
        self.band_weights = band_weights

        # Load CSV
        df = pd.read_csv(csv_path)

        # Safety checks
        required_cols = {"sideSlip", "band"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"CSV must contain columns {required_cols}, "
                f"but found {df.columns.tolist()}"
            )

        counts = df["count"].values.astype(float)

        if use_lds:
            counts = gaussian_filter1d(counts, sigma=lds_sigma)
        
        # Build lookup tables
        self.bins = np.array(sorted(df["sideSlip"].astype(float).unique()))
        self.bin_to_band = dict(
            zip(df["sideSlip"].astype(float), df["band"].astype(str))
        )

        new_bands = []
        for c in counts:
            if c >= 1000:
                new_bands.append("many")
            elif c >= 100:
                new_bands.append("medium")
            else:
                new_bands.append("few")

        df["band_lds"] = new_bands
        
        # Validate band_weights covers all bands
        csv_bands = set(df["band"].astype(str).unique())
        missing = csv_bands - set(band_weights.keys())
        if missing:
            raise ValueError(
                f"band_weights missing entries for: {missing}. "
                f"CSV contains bands: {csv_bands}"
            )

    def _nearest_bin(self, y):
        """
        Find the closest sideslip bin in the CSV
        (used if exact rounded value is not present).
        """
        idx = int(np.argmin(np.abs(self.bins - y)))
        return float(self.bins[idx])

    def band_of(self, y):
        """
        Returns 'many', 'medium', or 'few' for a given sideslip value.
        """
        y_round = round(float(y), self.round_to)

        if y_round in self.bin_to_band:
            return self.bin_to_band[y_round]

        # Fallback to nearest bin (robust to rounding noise)
        nearest = self._nearest_bin(y_round)
        return self.bin_to_band[nearest]

    def weight_of(self, y):
        """
        Returns the numeric weight for a given sideslip value.
        """
        band = self.band_of(y)
        return float(self.band_weights[band])
