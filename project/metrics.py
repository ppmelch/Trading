from libraries import *

# Performance metrics : Sharpe Ratio, Sortino Ratio, Calmar Ratio, Maximum Drawdown, Win Rate

def get_metrics (data:pd.DataFrame) -> float:
    #FALTA :
     # Rendimientos promedio
     # Std
     # Downside
     # Picos Max/Min
     # Operaciones ganadoras
     # Total de operaciones

    Ratio_Sharpe_diario = (rend_prom) / std * 100
    Ratio_Sharpe_mensual = (rend_prom) / std *  np.sqrt(21) * 100
    Ratio_Sharpe_anual = (rend_prom) / std *  np.sqrt(252) * 100

    Ratio_Sortino_diario = rend_prom / downside 
    Ratio_Sortino_mensual = rend_prom / downside *  np.sqrt(21) * 100
    Ratio_Sortino_anual = rend_prom / downside *  np.sqrt(252) * 100

    Max_Drawdown = (Picomax - valmin) / Picomax * 100

    Ratio_Calmar_diario = rend_prom / Max_Drawdown 
    Ratio_Calmar_mensual = rend_prom * 21 / Max_Drawdown 
    Ratio_Calmar_anual = rend_prom * 252/ Max_Drawdown  
    
    Win_Rate = (operaciones_win / Total_operaciones ) * 100


    R_Sharpe = [Ratio_Sharpe_diario, Ratio_Sharpe_mensual, Ratio_Sharpe_anual]
    R_Sortino = [Ratio_Sortino_diario, Ratio_Sortino_mensual, Ratio_Sortino_anual]
    R_Calmar = [Ratio_Calmar_diario, Ratio_Calmar_mensual, Ratio_Calmar_anual]
    Max_D = [Max_Drawdown, Max_Drawdown, Max_Drawdown]
    Win_Rate = [Win_Rate, Win_Rate, Win_Rate]

    metrics = pd.DataFrame(
        data = [R_Sharpe, R_Sortino, R_Calmar, Max_D, Win_Rate],
        index = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max Drawdown', 'Win Rate'],
        columns = ['Diario', 'Mensual', 'Anual']
    )

    return metrics


