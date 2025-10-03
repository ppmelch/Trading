
from libraries import *
from funtions import dateset_split
from metrics import Metrics


base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, "Binance_BTCUSDT_1h.csv")

data = pd.read_csv(file_path, skiprows=1).dropna()
data['Date'] = pd.to_datetime(data['Date'], format='mixed')

train, test, validation = dateset_split(data, 0.6, 0.2, 0.2)

metrics = Metrics(data.Close)
print("Data Sharpe after load:", metrics.sharpe)

train_dates = pd.concat([train['Date'], test['Date'].iloc[:1]]).tolist()
test_dates = pd.concat([train['Date'].iloc[-1:], test['Date']]).tolist()
valid_dates = pd.concat([test['Date'].iloc[-1:], validation['Date']]).tolist()
