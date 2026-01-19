
import pandas as pd

### MODEL DEVELOPMENT:
class HybridModelKarada: # 体
    def __init__(self, model_1, model_2, multioutput=False):
        self.model_1 = model_1
        self.model_2 = model_2
        self.y_columns = None  # store column names from fit method
        self.multioutput = multioutput

    def fit(self, X_1, X_2, y):
        # linear model for estimate trend and seasons:    
        self.model_1.fit(X_1, y)
        y_fit = pd.DataFrame(
            self.model_1.predict(X_1),
            index=X_1.index, columns=y.columns
        )
        # get residual from standard model
        y_resid = y - y_fit
        # model for estimate residuals:
        if self.multioutput:
            self.model_2.fit(X_2, y_resid)
        else:
            y_resid = y_resid.stack().squeeze() # wide to long
            self.model_2.fit(X_2, y_resid)
        
        self.y_columns = y.columns
        self.y_fit = y_fit
        self.y_resid = y_resid

    def predict(self, X_1, X_2):
        # predict trend and seasonality
        y_pred = pd.DataFrame(
            self.model_1.predict(X_1),
            index=X_1.index, columns=self.y_columns
        )
        # add residuals prediction:
        if self.multioutput:
            y_pred += self.model_2.predict(X_2)
        else:
            y_pred = y_pred.stack().squeeze()
            y_pred += self.model_2.predict(X_2)
        
        return y_pred