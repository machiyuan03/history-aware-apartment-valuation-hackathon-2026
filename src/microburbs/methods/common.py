import numpy as np

class MethodBase:
    """prepare/ adapt receive only information available at their cutoff.
    Context: history, cutoff, optional catalog and aligned train_rows;
    model artifacts are separately persisted. Temporal OOF candidates need train_rows.
    Unknown target labels never enter this API. Costs measured by orchestrator.
    """
    def prepare(self,X,y,context): raise NotImplementedError
    def adapt(self,X,y,context):
        # Default: online log-residual intercept; does not pretend to learn new columns.
        self.calibration=0.
        if len(y)>=30:
            raw=self.predict(X)
            self.calibration=float(np.clip(np.median(np.log(y)-np.log(raw)),-.2,.2))
        return self
    def finish(self,p):
        return np.asarray(p)*np.exp(getattr(self,'calibration',0.))

def prediction_vector(prediction, expected_rows):
    """One shared output boundary for every candidate; no silent reshape/broadcast."""
    values = np.asarray(prediction, dtype=float)
    if values.shape != (expected_rows,):
        raise ValueError(f'Expected prediction shape {(expected_rows,)}, got {values.shape}')
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Predictions must be finite positive prices')
    return values
