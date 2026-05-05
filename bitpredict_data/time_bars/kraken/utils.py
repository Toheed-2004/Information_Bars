def map_symbol(symbol: str) -> str:
    """
    Map generic symbol to Kraken futures symbol.
    """
    symbol = symbol.lower()

    symbols_map = {
            'btc': 'PF_XBTUSD', 'eth': 'PF_ETHUSD', 'sol': 'PF_SOLUSD', 'ada': 'PF_ADAUSD',
            'xrp': 'PF_XRPUSD', 'doge': 'PF_DOGEUSD', 'sui': 'PF_SUIUSD', 'ton': 'PF_TONUSD',
            'mina': 'PF_MINAUSD', 'ltc': 'PF_LTCUSD', 'bch': 'PF_BCHUSD', 'link': 'PF_LINKUSD',
            'dot': 'PF_DOTUSD', 'uni': 'PF_UNIUSD', 'aave': 'PF_AAVEUSD', 'crv': 'PF_CRVUSD',
            'comp': 'PF_COMPUSD', 'yfi': 'PF_YFIUSD', 'snx': 'PF_SNXUSD', 'mkr': 'PF_MKRUSD',
            'avax': 'PF_AVAXUSD', 'matic': 'PF_MATICUSD', 'bnb': 'PF_BNBUSD', 'etc': 'PF_ETCUSD',
            'atom': 'PF_ATOMUSD', 'algo': 'PF_ALGOUSD', 'sushi': 'PF_SUSHIUSD', '1inch': 'PF_1INCHUSD',
            'arb': 'PF_ARBUSD', 'op': 'PF_OPUSD', 'apt': 'PF_APTUSD', 'rndr': 'PF_RENDERUSD',
            'imx': 'PF_IMXUSD', 'ldo': 'PF_LDOUSD', 'paxg': 'PF_PAXGUSD', 'gmt': 'PF_GMTUSD',
            'sand': 'PF_SANDUSD', 'mana': 'PF_MANAUSD', 'gala': 'PF_GALAUSD', 'axs': 'PF_AXSUSD',
            'ape': 'PF_APEUSD', 'chz': 'PF_CHZUSD', 'enj': 'PF_ENJUSD', 'cfx': 'PF_CFXUSD',
            'neo': 'PF_NEOUSD', 'icp': 'PF_ICPUSD', 'rose': 'PF_ROSEUSD', 'hnt': 'PF_HNTUSD',
            'qnt': 'PF_QNTUSD', 'ftm': 'PF_FTMUSD', 'fil': 'PF_FILUSD', 'egld': 'PF_EGLDUSD',
            'zec': 'PF_ZECUSD', 'xmr': 'PF_XMRUSD', 'dash': 'PF_DASHUSD', 'xtz': 'PF_XTZUSD',
            'waves': 'PF_WAVESUSD', 'hbar': 'PF_HBARUSD', 'ksm': 'PF_KSMUSD', 'zil': 'PF_ZILUSD',
            'ont': 'PF_ONTUSD', 'iota': 'PF_IOTAUSD', 'bat': 'PF_BATUSDT', 'vet': 'PF_VETUSD',
            'qtum': 'PF_QTUMUSD', 'iost': 'PF_IOSTUSD', 'theta': 'PF_THETAUSD', 'knc': 'PF_KNCUSD',
            'zrx': 'PF_ZRXUSD', 'omg': 'PF_OMGUSD', 'sxp': 'PF_SXPUSD', 'kava': 'PF_KAVAUSD',
            'band': 'PF_BANDUSD', 'rlc': 'PF_RLCUSD', 'bal': 'PF_BALUSD', 'trb': 'PF_TRBUSD',
            'rune': 'PF_RUNEUSD', 'storj': 'PF_STORJUSD', 'blz': 'PF_BLZUSD', 'icx': 'PF_ICXUSD',
            'sc': 'PF_SCUSD', 'stx': 'PF_STXUSD', 'dydx': 'PF_DYDXUSD', '1000pepe': 'PF_1000PEPEUSD',
            '1000shib': 'PF_1000SHIBUSD', 'popcat': 'PF_POPCATUSD', 'meme': 'PF_MEMEUSD',
        }
    
    return symbols_map.get(symbol, f"PF_{symbol.upper()}USD")
