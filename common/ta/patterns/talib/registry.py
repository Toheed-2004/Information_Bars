from typing import Dict, List, Any


class PatternRegistry:
    """Complete registry containing ALL available TA-Lib candlestick patterns"""

    # ==============================================================================
    # TALIB CANDLESTICK PATTERNS (61 total)
    # ==============================================================================
    TALIB_PATTERNS = {
        "CDL2CROWS": {
            "lib": "talib",
            "func_name": "CDL2CROWS",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Two Crows",
        },
        "CDL3BLACKCROWS": {
            "lib": "talib",
            "func_name": "CDL3BLACKCROWS",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Three Black Crows",
        },
        "CDL3INSIDE": {
            "lib": "talib",
            "func_name": "CDL3INSIDE",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Three Inside Up/Down",
        },
        "CDL3LINESTRIKE": {
            "lib": "talib",
            "func_name": "CDL3LINESTRIKE",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Three-Line Strike",
        },
        "CDL3OUTSIDE": {
            "lib": "talib",
            "func_name": "CDL3OUTSIDE",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Three Outside Up/Down",
        },
        "CDL3STARSINSOUTH": {
            "lib": "talib",
            "func_name": "CDL3STARSINSOUTH",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Three Stars In The South",
        },
        "CDL3WHITESOLDIERS": {
            "lib": "talib",
            "func_name": "CDL3WHITESOLDIERS",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Three Advancing White Soldiers",
        },
        "CDLABANDONEDBABY": {
            "lib": "talib",
            "func_name": "CDLABANDONEDBABY",
            "inputs": ["open", "high", "low", "close"],
            "params": {
                "penetration": {"default": 0.3, "range": (0.0, 1.0), "type": float}
            },
            "outputs": ["pattern"],
            "description": "Abandoned Baby",
        },
        "CDLADVANCEBLOCK": {
            "lib": "talib",
            "func_name": "CDLADVANCEBLOCK",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Advance Block",
        },
        "CDLBELTHOLD": {
            "lib": "talib",
            "func_name": "CDLBELTHOLD",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Belt-hold",
        },
        "CDLBREAKAWAY": {
            "lib": "talib",
            "func_name": "CDLBREAKAWAY",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Breakaway",
        },
        "CDLCLOSINGMARUBOZU": {
            "lib": "talib",
            "func_name": "CDLCLOSINGMARUBOZU",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Closing Marubozu",
        },
        "CDLCONCEALBABYSWALL": {
            "lib": "talib",
            "func_name": "CDLCONCEALBABYSWALL",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Concealing Baby Swallow",
        },
        "CDLCOUNTERATTACK": {
            "lib": "talib",
            "func_name": "CDLCOUNTERATTACK",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Counterattack",
        },
        "CDLDARKCLOUDCOVER": {
            "lib": "talib",
            "func_name": "CDLDARKCLOUDCOVER",
            "inputs": ["open", "high", "low", "close"],
            "params": {
                "penetration": {"default": 0.5, "range": (0.0, 1.0), "type": float}
            },
            "outputs": ["pattern"],
            "description": "Dark Cloud Cover",
        },
        "CDLDOJI": {
            "lib": "talib",
            "func_name": "CDLDOJI",
            "category": "reversal",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Doji Pattern - Indecision pattern with small body, open and close are nearly equal",
            "column_examples": ["pattern_doji"],
            "interpretation": "100=bullish doji, -100=bearish doji, 0=no pattern. Signals market indecision and potential reversal.",
            "characteristics": "Small body (open ≈ close), often long upper and/or lower shadows. Found at trend extremes.",
        },
        "CDLDOJISTAR": {
            "lib": "talib",
            "func_name": "CDLDOJISTAR",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Doji Star",
        },
        "CDLDRAGONFLYDOJI": {
            "lib": "talib",
            "func_name": "CDLDRAGONFLYDOJI",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Dragonfly Doji",
        },
        "CDLENGULFING": {
            "lib": "talib",
            "func_name": "CDLENGULFING",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Engulfing Pattern",
        },
        "CDLEVENINGDOJISTAR": {
            "lib": "talib",
            "func_name": "CDLEVENINGDOJISTAR",
            "inputs": ["open", "high", "low", "close"],
            "params": {
                "penetration": {"default": 0.3, "range": (0.0, 1.0), "type": float}
            },
            "outputs": ["pattern"],
            "description": "Evening Doji Star",
        },
        "CDLEVENINGSTAR": {
            "lib": "talib",
            "func_name": "CDLEVENINGSTAR",
            "inputs": ["open", "high", "low", "close"],
            "params": {
                "penetration": {"default": 0.3, "range": (0.0, 1.0), "type": float}
            },
            "outputs": ["pattern"],
            "description": "Evening Star",
        },
        "CDLGAPSIDESIDEWHITE": {
            "lib": "talib",
            "func_name": "CDLGAPSIDESIDEWHITE",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Up/Down-gap side-by-side white lines",
        },
        "CDLGRAVESTONEDOJI": {
            "lib": "talib",
            "func_name": "CDLGRAVESTONEDOJI",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Gravestone Doji",
        },
        "CDLHAMMER": {
            "lib": "talib",
            "func_name": "CDLHAMMER",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Hammer",
        },
        "CDLHANGINGMAN": {
            "lib": "talib",
            "func_name": "CDLHANGINGMAN",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Hanging Man",
        },
        "CDLHARAMI": {
            "lib": "talib",
            "func_name": "CDLHARAMI",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Harami Pattern",
        },
        "CDLHARAMICROSS": {
            "lib": "talib",
            "func_name": "CDLHARAMICROSS",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Harami Cross Pattern",
        },
        "CDLHIGHWAVE": {
            "lib": "talib",
            "func_name": "CDLHIGHWAVE",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "High-Wave Candle",
        },
        "CDLHIKKAKE": {
            "lib": "talib",
            "func_name": "CDLHIKKAKE",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Hikkake Pattern",
        },
        "CDLHIKKAKEMOD": {
            "lib": "talib",
            "func_name": "CDLHIKKAKEMOD",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Modified Hikkake Pattern",
        },
        "CDLHOMINGPIGEON": {
            "lib": "talib",
            "func_name": "CDLHOMINGPIGEON",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Homing Pigeon",
        },
        "CDLIDENTICAL3CROWS": {
            "lib": "talib",
            "func_name": "CDLIDENTICAL3CROWS",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Identical Three Crows",
        },
        "CDLINNECK": {
            "lib": "talib",
            "func_name": "CDLINNECK",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "In-Neck Pattern",
        },
        "CDLINVERTEDHAMMER": {
            "lib": "talib",
            "func_name": "CDLINVERTEDHAMMER",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Inverted Hammer",
        },
        "CDLKICKING": {
            "lib": "talib",
            "func_name": "CDLKICKING",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Kicking",
        },
        "CDLKICKINGBYLENGTH": {
            "lib": "talib",
            "func_name": "CDLKICKINGBYLENGTH",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Kicking - bull/bear determined by the longer marubozu",
        },
        "CDLLADDERBOTTOM": {
            "lib": "talib",
            "func_name": "CDLLADDERBOTTOM",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Ladder Bottom",
        },
        "CDLLONGLEGGEDDOJI": {
            "lib": "talib",
            "func_name": "CDLLONGLEGGEDDOJI",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Long Legged Doji",
        },
        "CDLLONGLINE": {
            "lib": "talib",
            "func_name": "CDLLONGLINE",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Long Line Candle",
        },
        "CDLMARUBOZU": {
            "lib": "talib",
            "func_name": "CDLMARUBOZU",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Marubozu",
        },
        "CDLMATCHINGLOW": {
            "lib": "talib",
            "func_name": "CDLMATCHINGLOW",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Matching Low",
        },
        "CDLMATHOLD": {
            "lib": "talib",
            "func_name": "CDLMATHOLD",
            "inputs": ["open", "high", "low", "close"],
            "params": {
                "penetration": {"default": 0.5, "range": (0.0, 1.0), "type": float}
            },
            "outputs": ["pattern"],
            "description": "Mat Hold",
        },
        "CDLMORNINGDOJISTAR": {
            "lib": "talib",
            "func_name": "CDLMORNINGDOJISTAR",
            "inputs": ["open", "high", "low", "close"],
            "params": {
                "penetration": {"default": 0.3, "range": (0.0, 1.0), "type": float}
            },
            "outputs": ["pattern"],
            "description": "Morning Doji Star",
        },
        "CDLMORNINGSTAR": {
            "lib": "talib",
            "func_name": "CDLMORNINGSTAR",
            "inputs": ["open", "high", "low", "close"],
            "params": {
                "penetration": {"default": 0.3, "range": (0.0, 1.0), "type": float}
            },
            "outputs": ["pattern"],
            "description": "Morning Star",
        },
        "CDLONNECK": {
            "lib": "talib",
            "func_name": "CDLONNECK",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "On-Neck Pattern",
        },
        "CDLPIERCING": {
            "lib": "talib",
            "func_name": "CDLPIERCING",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Piercing Pattern",
        },
        "CDLRICKSHAWMAN": {
            "lib": "talib",
            "func_name": "CDLRICKSHAWMAN",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Rickshaw Man",
        },
        "CDLRISEFALL3METHODS": {
            "lib": "talib",
            "func_name": "CDLRISEFALL3METHODS",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Rising/Falling Three Methods",
        },
        "CDLSEPARATINGLINES": {
            "lib": "talib",
            "func_name": "CDLSEPARATINGLINES",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Separating Lines",
        },
        "CDLSHOOTINGSTAR": {
            "lib": "talib",
            "func_name": "CDLSHOOTINGSTAR",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Shooting Star",
        },
        "CDLSHORTLINE": {
            "lib": "talib",
            "func_name": "CDLSHORTLINE",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Short Line Candle",
        },
        "CDLSPINNINGTOP": {
            "lib": "talib",
            "func_name": "CDLSPINNINGTOP",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Spinning Top",
        },
        "CDLSTALLEDPATTERN": {
            "lib": "talib",
            "func_name": "CDLSTALLEDPATTERN",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Stalled Pattern",
        },
        "CDLSTICKSANDWICH": {
            "lib": "talib",
            "func_name": "CDLSTICKSANDWICH",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Stick Sandwich",
        },
        "CDLTAKURI": {
            "lib": "talib",
            "func_name": "CDLTAKURI",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Takuri (Dragonfly Doji with very long lower shadow)",
        },
        "CDLTASUKIGAP": {
            "lib": "talib",
            "func_name": "CDLTASUKIGAP",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Tasuki Gap",
        },
        "CDLTHRUSTING": {
            "lib": "talib",
            "func_name": "CDLTHRUSTING",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Thrusting Pattern",
        },
        "CDLTRISTAR": {
            "lib": "talib",
            "func_name": "CDLTRISTAR",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Tristar Pattern",
        },
        "CDLUNIQUE3RIVER": {
            "lib": "talib",
            "func_name": "CDLUNIQUE3RIVER",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Unique 3 River",
        },
        "CDLUPSIDEGAP2CROWS": {
            "lib": "talib",
            "func_name": "CDLUPSIDEGAP2CROWS",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Upside Gap Two Crows",
        },
        "CDLXSIDEGAP3METHODS": {
            "lib": "talib",
            "func_name": "CDLXSIDEGAP3METHODS",
            "inputs": ["open", "high", "low", "close"],
            "params": {},
            "outputs": ["pattern"],
            "description": "Upside/Downside Gap Three Methods",
        },
    }

    # ==============================================================================
    # UTILITY METHODS
    # ==============================================================================
    @classmethod
    def get_all_patterns(cls) -> Dict[str, Dict]:
        """Get all available TALib patterns"""
        return cls.TALIB_PATTERNS

    @classmethod
    def get_all(cls) -> Dict[str, Dict]:
        """Get all patterns"""
        return cls.TALIB_PATTERNS

    @classmethod
    def get_pattern(cls, name: str) -> Dict:
        """Get specific pattern definition"""
        if name not in cls.TALIB_PATTERNS:
            raise ValueError(f"Pattern '{name}' not found")
        return cls.TALIB_PATTERNS[name]

    @classmethod
    def get_pattern_info(cls, pattern_name: str) -> Dict[str, Any]:
        """Get information about a specific pattern"""
        return cls.get_pattern(pattern_name)

    @classmethod
    def get_pattern_description(cls, pattern_name: str) -> str:
        """Get description of a pattern"""
        pattern_def = cls.get_pattern(pattern_name)
        return pattern_def.get("description", "")

    @classmethod
    def get_column_examples(cls, pattern_name: str) -> List[str]:
        """Get column naming examples for a pattern"""
        pattern_def = cls.get_pattern(pattern_name)
        return pattern_def.get("column_examples", [])

    @classmethod
    def get_interpretation_guide(cls, pattern_name: str) -> str:
        """Get interpretation guide for a pattern"""
        pattern_def = cls.get_pattern(pattern_name)
        return pattern_def.get("interpretation", "")

    @classmethod
    def validate_params(cls, pattern_name: str, params: Dict) -> bool:
        """Validate parameter values for a pattern"""
        pattern_def = cls.get_pattern(pattern_name)

        for param_name, param_value in params.items():
            if param_name not in pattern_def["params"]:
                raise ValueError(
                    f"Unknown parameter '{param_name}' for pattern '{pattern_name}'"
                )

            param_spec = pattern_def["params"][param_name]

            # Handle default value extraction
            if isinstance(param_spec, dict) and "default" in param_spec:
                # This is the metadata format: {'default': 0.3, 'range': (0.0, 1.0), 'type': float}
                # Extract just the default value for validation
                default_value = param_spec["default"]

                # Type validation if type is specified
                if "type" in param_spec:
                    expected_type = param_spec["type"]
                    if not isinstance(param_value, expected_type):
                        raise ValueError(
                            f"Parameter '{param_name}' must be of type {expected_type.__name__}"
                        )

                # Range validation if range is specified
                if "range" in param_spec and param_value is not None:
                    min_val, max_val = param_spec["range"]
                    if not (min_val <= param_value <= max_val):
                        raise ValueError(
                            f"Parameter '{param_name}' must be between {min_val} and {max_val}"
                        )
            else:
                # Direct value comparison (for patterns without metadata)
                if param_value != param_spec:
                    raise ValueError(f"Parameter '{param_name}' must be {param_spec}")

        return True

    @classmethod
    def is_pattern_available(cls, pattern_name: str) -> bool:
        """Check if a pattern is available"""
        return pattern_name in cls.TALIB_PATTERNS
