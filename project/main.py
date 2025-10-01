from libraries import *
from funtions import dateset_split

def main():

    data = pd.read_csv('Binance_BTCUSDT_1h.csv', skiprows=1).dropna()
    data['Date'] = pd.to_datetime(data['Date'], format='mixed')

    train, test, validation = dateset_split(data, 0.6, 0.2, 0.2)

if __name__ == "__main__":
    main()