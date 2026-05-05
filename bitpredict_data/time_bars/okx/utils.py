def map_symbol(symbol: str) -> str:
    """
    Map generic symbol to OKX swap symbol.
    """
    symbol = symbol.lower()

    symbols_map = {
            'btc': 'BTC-USDT-SWAP', 'eth': 'ETH-USDT-SWAP', 'sol': 'SOL-USDT-SWAP',
            'ada': 'ADA-USDT-SWAP', 'xrp': 'XRP-USDT-SWAP', 'doge': 'DOGE-USDT-SWAP',
            'sui': 'SUI-USDT-SWAP', 'ton': 'TON-USDT-SWAP', 'mina': 'MINA-USDT-SWAP',
            'ltc': 'LTC-USDT-SWAP', 'bch': 'BCH-USDT-SWAP', 'link': 'LINK-USDT-SWAP',
            'dot': 'DOT-USDT-SWAP', 'uni': 'UNI-USDT-SWAP', 'aave': 'AAVE-USDT-SWAP',
            'crv': 'CRV-USDT-SWAP', 'comp': 'COMP-USDT-SWAP', 'yfi': 'YFI-USDT-SWAP',
            'snx': 'SNX-USDT-SWAP', 'mkr': 'MKR-USDT-SWAP', 'avax': 'AVAX-USDT-SWAP',
            'matic': 'MATIC-USDT-SWAP', 'bnb': 'BNB-USDT-SWAP', 'etc': 'ETC-USDT-SWAP',
            'atom': 'ATOM-USDT-SWAP', 'algo': 'ALGO-USDT-SWAP', 'sushi': 'SUSHI-USDT-SWAP',
            '1inch': '1INCH-USDT-SWAP', 'arb': 'ARB-USDT-SWAP', 'op': 'OP-USDT-SWAP',
            'apt': 'APT-USDT-SWAP', 'rndr': 'RENDER-USDT-SWAP', 'imx': 'IMX-USDT-SWAP',
            'ldo': 'LDO-USDT-SWAP', 'paxg': 'PAXG-USDT-SWAP', 'gmt': 'GMT-USDT-SWAP',
            'sand': 'SAND-USDT-SWAP', 'mana': 'MANA-USDT-SWAP', 'gala': 'GALA-USDT-SWAP',
            'axs': 'AXS-USDT-SWAP', 'ape': 'APE-USDT-SWAP', 'chz': 'CHZ-USDT-SWAP',
            'enj': 'ENJ-USDT-SWAP', 'cfx': 'CFX-USDT-SWAP', 'neo': 'NEO-USDT-SWAP',
            'icp': 'ICP-USDT-SWAP', 'rose': 'ROSE-USDT-SWAP', 'hnt': 'HNT-USDT-SWAP',
            'qnt': 'QNT-USDT-SWAP', 'ftm': 'FTM-USDT-SWAP', 'fil': 'FIL-USDT-SWAP',
            'egld': 'EGLD-USDT-SWAP', 'zec': 'ZEC-USDT-SWAP', 'xmr': 'XMR-USDT-SWAP',
            'dash': 'DASH-USDT-SWAP', 'xtz': 'XTZ-USDT-SWAP', 'waves': 'WAVES-USDT-SWAP',
            'hbar': 'HBAR-USDT-SWAP', 'ksm': 'KSM-USDT-SWAP', 'zil': 'ZIL-USDT-SWAP',
            'ont': 'ONT-USDT-SWAP', 'iota': 'IOTA-USDT-SWAP', 'bat': 'BAT-USDT-SWAP',
            'vet': 'VET-USDT-SWAP', 'qtum': 'QTUM-USDT-SWAP', 'iost': 'IOST-USDT-SWAP',
            'theta': 'THETA-USDT-SWAP', 'knc': 'KNC-USDT-SWAP', 'zrx': 'ZRX-USDT-SWAP',
            'omg': 'OMG-USDT-SWAP', 'sxp': 'SXP-USDT-SWAP', 'kava': 'KAVA-USDT-SWAP',
            'band': 'BAND-USDT-SWAP', 'rlc': 'RLC-USDT-SWAP', 'bal': 'BAL-USDT-SWAP',
            'trb': 'TRB-USDT-SWAP', 'rune': 'RUNE-USDT-SWAP', 'storj': 'STORJ-USDT-SWAP',
            'blz': 'BLZ-USDT-SWAP', 'icx': 'ICX-USDT-SWAP', 'sc': 'SC-USDT-SWAP',
            'stx': 'STX-USDT-SWAP', 'dydx': 'DYDX-USDT-SWAP', '1000pepe': '1000PEPE-USDT-SWAP',
            '1000shib': '1000SHIB-USDT-SWAP', 'popcat': 'POPCAT-USDT-SWAP', 'meme': 'MEME-USDT-SWAP',
        }

    return symbols_map.get(symbol, f"{symbol.upper()}-USDT-SWAP")
