#!/bin/python3
"""Entry script."""

import time

from modules.instruments import Portfolio
from modules.server import Server
from modules.algorithms import Hedge
from modules import experts

# S&P 500 tickers
STOCKS = [
    "AAPL", "MSFT", "AMZN", "GOOG", "GOOGL", "FB", "TSLA", "BRK", "V", "JPM", "JNJ", "WMT", "NVDA", "PYPL", "MA", "DIS", "PG", "UNH", "HD", "BAC", "INTC", "NFLX", "CMCSA", "ADBE", "CRM", "ABT", "VZ", "NKE", "XOM", "KO", "T", "AVGO", "TMO", "LLY", "CSCO", "PFE", "MRK", "PEP", "ABBV", "ORCL", "CVX", "DHR", "ACN", "QCOM", "TXN", "MDT", "MCD", "NEE", "COST", "TMUS",
    "WFC", "UNP", "HON", "UPS", "MS", "AMGN", "PM", "C", "BMY", "LIN", "LOW", "BA", "SBUX", "CHTR", "INTU", "NOW", "SCHW", "BLK", "AMD", "RTX", "CAT", "AMAT", "GS", "EL", "IBM", "AXP", "GE", "MMM", "AMT", "DE", "MU", "TGT", "LMT", "SYK", "ISRG", "CVS", "BKNG", "LRCX", "FIS", "SPGI", "GILD", "TJX", "MO", "ATVI", "ZTS", "PLD", "MDLZ", "GM", "TFC", "BDX",
    "CB", "FISV", "USB", "PNC", "ANTM", "ILMN", "CI", "ADP", "CCI", "FDX", "CSX", "CME", "ADSK", "CL", "COP", "DUK", "NSC", "SHW", "ICE", "ITW", "SO", "EQIX", "ECL", "ADI", "GPN", "HCA", "TWTR", "MMC", "D", "APD", "COF", "VRTX", "EW", "BSX", "MCO", "KLAC", "AON", "REGN", "EMR", "MET", "ETN", "PGR", "HUM", "DG", "MNST", "NOC", "ALGN", "FCX", "WM", "GD",
    "NEM", "IDXX", "F", "STZ", "SNPS", "KMB", "LVS", "DOW", "MCHP", "TEL", "KHC", "EBAY", "ROST", "BIIB", "WBA", "MAR", "EA", "APTV", "CMG", "EXC", "CDNS", "APH", "CTSH", "ROP", "PSA", "BAX", "A", "SYY", "DXCM", "AEP", "LHX", "DLR", "DD", "JCI", "BK", "SLB", "TROW", "TRV", "INFO", "EOG", "VIAC", "IQV", "MSCI", "AIG", "CTAS", "CMI", "SPG", "TT", "SRE", "PH",
    "BF", "HPQ", "XLNX", "PSX", "ANSS", "ALXN", "GIS", "KMI", "IFF", "MPC", "CNC", "CTVA", "PCAR", "ZBH", "AFL", "PRU", "LYB", "XEL", "PPG", "CARR", "PAYX", "SWKS", "YUM", "VFC", "HLT", "HSY", "ALL", "BBY", "ORLY", "ADM", "TDG", "MSI", "LUV", "VRSK", "GLW", "BLL", "DFS", "PXD", "WLTW", "PEG", "AWK", "ROK", "SBAC", "RSG", "ETSY", "DHI", "MCK", "ES", "DAL", "MTD",
    "RMD", "KEYS", "FRC", "WELL", "CPRT", "AME", "WMB", "SWK", "OTIS", "VLO", "SIVB", "LEN", "FAST", "FTNT", "AZO", "STT", "WY", "ZBRA", "MXIM", "AMP", "KR", "WEC", "HRL", "GRMN", "DLTR", "CCL", "EQR", "AVB", "ODFL", "OXY", "ENPH", "ANET", "BKR", "TSN", "FITB", "TER", "ED", "CBRE", "NDAQ", "FTV", "DTE", "ARE", "MKC", "CLX", "O", "LH", "AJG", "VRSN", "FLT", "TTWO",
    "PAYC", "CERN", "CDW", "VTRS", "SYF", "WST", "EIX", "PPL", "VMC", "HOLX", "EXPE", "EFX", "ABC", "DISCK", "DISCA", "CTLT", "URI", "NTRS", "OKE", "MKTX", "WDC", "MLM", "QRVO", "CHD", "KMX", "GWW", "K", "RF", "IP", "KEY", "BIO", "AES", "KSU", "MTB", "TYL", "COO", "HES", "ALB", "TSCO", "VTR", "TFX", "TRMB", "ETR", "FOX", "FOXA", "HPE", "ULTA", "HAL", "IR", "CFG",
    "LYV",
    "ROL", "INCY", "AEE", "XYL", "RCL", "AMCR", "FE", "WAT", "MPWR", "HIG", "ESS", "DOV", "NUE", "MGM", "NVR", "DISH", "BR", "CTXS", "STX", "DRI", "CAG", "DGX", "PKI", "PEAK", "RJF", "VAR", "AKAM", "CMS", "IT", "MAA", "EXPD", "STE", "JBHT", "DRE", "EXR", "NTAP", "CE", "HBAN", "WAB", "CAH", "LDOS", "EMN", "AVY", "IEX", "DPZ", "PFG", "J", "GPC", "CINF", "ABMD",
    "BXP", "TDY", "UAL", "BEN", "OMC", "FMC", "NWSA", "DVN", "WYNN", "CPB", "NWS", "MAS", "LB", "IPGP", "POOL", "L", "LUMN", "FFIV", "SJM", "PKG", "UDR", "NLOK", "PHM", "WHR", "HAS", "EVRG", "HWM", "MHK", "WRB", "CHRW", "FBHS", "LNT", "XRAY", "TXT", "CNP", "ATO", "MOS", "WRK", "LKQ", "DVA", "LW", "JKHY", "AAL", "HST", "FANG", "UHS", "TPR", "PWR", "BWA", "AAP",
    "IVZ", "LNC", "SNA", "ALLE", "NWL", "HSIC", "NRG", "WU", "GL", "IPG", "CF", "TAP", "RE", "AOS", "IRM", "UAA", "UA", "CMA", "PNR", "REG", "GPS", "NI", "PNW", "RHI", "ZION", "NLSN", "RL", "JNPR", "NCLH", "FRT", "KIM", "MRO", "ALK", "AIZ", "COG", "FLIR", "VNO", "HII", "APA", "PVH", "SEE", "PBCT", "DXC", "HBI", "PRGO", "NOV", "LEG", "VNT", "HFC", "UNM",
    "FLS", "XRX", "SLG",
]


if __name__ == "__main__":
    srv = Server()
    srv.start()

    # TODO: do it non-simulated
    with srv.simulated():
        # retrieve stocks
        start_t = time.perf_counter()
        stocks = srv[STOCKS, '1y', '1d', dict(ignore_errors=True)]
        end_t = time.perf_counter()
        print(f'Stock request time: {end_t-start_t:.2f}s')

        # create 100 USD portfolio
        # TODO: portfolio and hedge shouldn't be created each time, but saved and loaded from disk
        pf = Portfolio(wealth=100, stocks=stocks)

        srv.add_portfolio(pf)

        # ensemble algorithm
        hedge = Hedge(srv, pf, stocks[0].bar)

        # add sharpe ratio maximizer expert
        hedge.add(experts.Sharpe)

        hedge.start()
