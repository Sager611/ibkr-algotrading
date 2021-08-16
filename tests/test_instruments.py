from modules.instruments import Stock


def test_init():
    info_data = [
        {
            "conid": "756733",
            "symbol": "SPY",
            "companyHeader": "SPDR S&P 500 ETF TRUST - ARCA",
            "companyName": "SPDR S&P 500 ETF TRUST"
        }
    ]

    hist_data = {
        "data": []
    }

    stk = Stock(info_data=info_data, hist_data=hist_data)

    assert stk.conid == info_data[0]["conid"]
    assert stk.symbol == info_data[0]["symbol"]
    assert stk.companyHeader == info_data[0]["companyHeader"]
    assert stk.companyName == info_data[0]["companyName"]
