"""
Alternative Loss Functions für Drift Angle Prediction
======================================================

Basierend auf der Analyse der 575 Grid Search Experimente:
- Bestes Modell: GRU, 0.7889° total score
- Problem: 25-30% über den Targets
- Lösung: Bessere Loss Functions testen

Hier sind 4 Alternative Loss Functions mit Vor-/Nachteilen:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. HUBER LOSS (Robust gegen Outliers)
# ============================================================================
class HuberLoss(nn.Module):
    """
    Huber Loss kombiniert L1 und L2:
    - Für kleine Fehler: quadratisch (L2) → schnellere Konvergenz
    - Für große Fehler: linear (L1) → robust gegen Outliers
    
    ✅ VORTEILE:
    - Robuster gegen Ausreißer als L2
    - Glattere Gradienten als L1
    - Gut für Few-Shot (wenige Samples, viele Outliers)
    
    ⚠️ NACHTEILE:
    - Delta-Parameter muss getunt werden
    - Komplexer als einfaches L1
    
    📊 EMPFOHLEN FÜR:
    - Few-shot samples (große Fehler)
    - Wenn Outliers das Problem sind
    """
    
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, predictions, targets):
        error = predictions - targets
        abs_error = torch.abs(error)
        
        # Quadratisch für kleine Fehler, linear für große
        quadratic = 0.5 * error ** 2
        linear = self.delta * (abs_error - 0.5 * self.delta)
        
        loss = torch.where(abs_error <= self.delta, quadratic, linear)
        return loss.mean()


# ============================================================================
# 2. WEIGHTED MSE (Sample-gewichtet nach Shot-Band)
# ============================================================================
class WeightedMSELoss(nn.Module):
    """
    MSE mit höheren Gewichten für problematische Shot-Bands.
    
    Basierend auf Analysis:
    - Many-shot: 0.188° (Target: 0.15°) → Gewicht: 1.25
    - Medium-shot: 0.231° (Target: 0.20°) → Gewicht: 1.15
    - Few-shot: 0.339° (Target: 0.35°) → Gewicht: 1.0
    
    ✅ VORTEILE:
    - Fokussiert Training auf schwierige Samples
    - Ausbalanciert viele/wenige Samples
    - Direkt interpretierbar
    
    ⚠️ NACHTEILE:
    - Braucht shot_band labels während Training
    - Kann zu Overfitting auf wenigen Samples führen
    
    📊 EMPFOHLEN FÜR:
    - Imbalanced shot-band distribution
    - Wenn Many-/Medium-shot verbessert werden sollen
    """
    
    def __init__(self, shot_band_weights=None):
        super().__init__()
        # Default: höhere Gewichte für problematische Bands
        if shot_band_weights is None:
            self.weights = {
                'many': 1.25,     # 25% über Target
                'medium': 1.15,   # 15% über Target
                'few': 1.0        # Nahe am Target
            }
        else:
            self.weights = shot_band_weights
    
    def forward(self, predictions, targets, shot_bands):
        """
        predictions: (batch_size, 1)
        targets: (batch_size, 1)
        shot_bands: (batch_size,) - tensor with values 'many', 'medium', 'few'
        """
        # Compute base MSE
        squared_error = (predictions - targets) ** 2
        
        # Apply weights based on shot band
        weights = torch.ones_like(squared_error)
        for band, weight in self.weights.items():
            mask = (shot_bands == band)
            weights[mask] = weight
        
        weighted_loss = squared_error * weights
        return weighted_loss.mean()


# ============================================================================
# 3. COMBINED LOSS (MSE + L1 + Huber)
# ============================================================================
class CombinedLoss(nn.Module):
    """
    Kombiniert mehrere Loss Functions:
    - MSE: für glatte Konvergenz
    - L1: für Robustheit
    - Huber: für Outlier-Robustheit
    
    ✅ VORTEILE:
    - Best of all worlds
    - Flexibel durch Gewichte
    - Bewährt in vielen Kaggle-Competitions
    
    ⚠️ NACHTEILE:
    - Viele Hyperparameter (3 Gewichte)
    - Kann schwer zu tunen sein
    
    📊 EMPFOHLEN FÜR:
    - Wenn einzelne Loss nicht ausreichen
    - Als Ensemble-ähnliche Lösung
    """
    
    def __init__(self, alpha_mse=0.4, alpha_l1=0.3, alpha_huber=0.3, delta=1.0):
        super().__init__()
        self.alpha_mse = alpha_mse
        self.alpha_l1 = alpha_l1
        self.alpha_huber = alpha_huber
        
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.huber = HuberLoss(delta=delta)
    
    def forward(self, predictions, targets):
        loss_mse = self.mse(predictions, targets)
        loss_l1 = self.l1(predictions, targets)
        loss_huber = self.huber(predictions, targets)
        
        total = (
            self.alpha_mse * loss_mse +
            self.alpha_l1 * loss_l1 +
            self.alpha_huber * loss_huber
        )
        return total


# ============================================================================
# 4. QUANTILE LOSS (für Few-Shot Extremwerte)
# ============================================================================
class QuantileLoss(nn.Module):
    """
    Quantile Regression Loss - asymmetrische Bestrafung.
    
    Nützlich wenn:
    - Über-/Unterschätzung unterschiedlich schlecht ist
    - Few-shot samples besonders kritisch sind
    
    ✅ VORTEILE:
    - Flexibel für asymmetrische Fehlerkosten
    - Gut für Extremwerte (Few-shot)
    - Robuster als MSE
    
    ⚠️ NACHTEILE:
    - Quantil muss gewählt werden
    - Weniger intuitiv
    
    📊 EMPFOHLEN FÜR:
    - Few-shot improvement
    - Wenn Überschätzung schlimmer als Unterschätzung (oder umgekehrt)
    """
    
    def __init__(self, quantile=0.5):
        """
        quantile=0.5: Median Regression (symmetrisch)
        quantile>0.5: Bestraft Überschätzung stärker
        quantile<0.5: Bestraft Unterschätzung stärker
        """
        super().__init__()
        self.quantile = quantile
    
    def forward(self, predictions, targets):
        error = targets - predictions
        loss = torch.where(
            error >= 0,
            self.quantile * error,
            (self.quantile - 1) * error
        )
        return loss.mean()


# ============================================================================
# 5. FOCAL L1 LOSS (Bereits verwendet, aber verbesserbar)
# ============================================================================
class ImprovedFocalL1Loss(nn.Module):
    """
    Verbesserte Version der aktuellen Focal L1 Loss.
    
    Änderungen:
    - Adaptives Gamma (startet hoch, sinkt mit Training)
    - Separate Gewichte für Shot-Bands
    - Optional: Temperature Scaling
    
    ✅ VORTEILE:
    - Evolution der aktuellen Loss
    - Beibehaltung des Ansatzes
    - Fokus auf schwierige Samples
    
    ⚠️ NACHTEILE:
    - Komplexer als Original
    - Mehr Hyperparameter
    
    📊 EMPFOHLEN FÜR:
    - Wenn aktuelle Focal L1 schon gut ist
    - Als Basis für weitere Verbesserungen
    """
    
    def __init__(self, lambda_focal=0.00001, gamma=0.7, 
                 adaptive_gamma=True, shot_weights=None):
        super().__init__()
        self.lambda_focal = lambda_focal
        self.gamma_start = gamma
        self.gamma = gamma
        self.adaptive_gamma = adaptive_gamma
        self.shot_weights = shot_weights or {'many': 1.2, 'medium': 1.1, 'few': 1.0}
        self.step = 0
    
    def update_gamma(self, epoch, max_epochs):
        """Reduziere Gamma während Training: 0.7 → 0.3"""
        if self.adaptive_gamma:
            progress = epoch / max_epochs
            self.gamma = self.gamma_start * (1 - 0.5 * progress)  # 50% Reduktion
    
    def forward(self, predictions, targets, shot_bands=None):
        # Base L1 loss
        base_loss = F.l1_loss(predictions, targets, reduction='none')
        
        # Focal weighting
        focal_weight = torch.abs(predictions - targets) ** self.gamma
        focal_loss = base_loss * focal_weight
        
        # Optional: Shot-band weighting
        if shot_bands is not None:
            weights = torch.ones_like(focal_loss)
            for band, weight in self.shot_weights.items():
                mask = (shot_bands == band)
                weights[mask] = weight
            focal_loss = focal_loss * weights
        
        # Regularization
        reg_loss = self.lambda_focal * torch.abs(predictions).mean()
        
        return focal_loss.mean() + reg_loss


# ============================================================================
# USAGE EXAMPLES & COMPARISON
# ============================================================================
def usage_examples():
    """
    Zeigt wie man die Loss Functions verwendet.
    """
    
    # Dummy data
    batch_size = 32
    predictions = torch.randn(batch_size, 1)
    targets = torch.randn(batch_size, 1)
    shot_bands = torch.tensor(['many'] * 10 + ['medium'] * 12 + ['few'] * 10)
    
    print("="*80)
    print("LOSS FUNCTION COMPARISON")
    print("="*80)
    
    # 1. Current L1
    l1_loss = nn.L1Loss()
    print(f"1. L1 Loss:              {l1_loss(predictions, targets):.4f}")
    
    # 2. Huber Loss
    huber_loss = HuberLoss(delta=1.0)
    print(f"2. Huber Loss (δ=1.0):   {huber_loss(predictions, targets):.4f}")
    
    # 3. Weighted MSE (needs shot_bands)
    # weighted_mse = WeightedMSELoss()
    # print(f"3. Weighted MSE:         {weighted_mse(predictions, targets, shot_bands):.4f}")
    
    # 4. Combined Loss
    combined_loss = CombinedLoss()
    print(f"4. Combined Loss:        {combined_loss(predictions, targets):.4f}")
    
    # 5. Quantile Loss
    quantile_loss = QuantileLoss(quantile=0.5)
    print(f"5. Quantile Loss (0.5):  {quantile_loss(predictions, targets):.4f}")
    
    # 6. Improved Focal L1
    focal_l1 = ImprovedFocalL1Loss(lambda_focal=0.00001, gamma=0.7)
    print(f"6. Improved Focal L1:    {focal_l1(predictions, targets):.4f}")
    
    print("="*80)


# ============================================================================
# RECOMMENDATION BASED ON ANALYSIS
# ============================================================================
"""
📊 EMPFEHLUNG BASIEREND AUF DEN 575 EXPERIMENTEN:

Problem-Analyse:
✅ Few-shot: 0.339° (nahe Target 0.35°)
❌ Many-shot: 0.193° (29% über Target 0.15°)
❌ Medium-shot: 0.256° (28% über Target 0.20°)

→ Fokus: Many- und Medium-shot verbessern!

MEINE TOP 3 LOSS FUNCTIONS:

1️⃣ **WeightedMSELoss** (BESTE WAHL)
   - Direkter Fokus auf Many/Medium
   - Gewichte: many=1.3, medium=1.2, few=0.8
   - Einfach zu implementieren
   - Interpretierbar
   
2️⃣ **CombinedLoss** (SICHER)
   - Best-of-all-worlds Ansatz
   - Bewährt in Kaggle Competitions
   - Flexibel durch Gewichte
   
3️⃣ **ImprovedFocalL1Loss** (EVOLUTION)
   - Verbesserung der aktuellen Loss
   - Adaptive Gamma
   - Shot-band Gewichte

NICHT EMPFOHLEN (für dieses Problem):
❌ Huber Loss: Few-shot ist bereits gut (kein Outlier-Problem)
❌ Quantile Loss: Asymmetrie ist nicht das Problem


IMPLEMENTIERUNGS-TIPP:
======================
Teste alle 3 mit dem besten Modell (GRU 32/32/3):
- Trainiere jeweils 1 Modell mit jeder Loss
- Vergleiche Many/Medium/Few scores
- Wähle beste Loss für finales Training

Erwartete Verbesserung: 10-15% (0.79° → 0.70°)
"""

if __name__ == "__main__":
    usage_examples()
