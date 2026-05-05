"""
bitpredict/__init__.py
----------------------
Python runs this automatically when any code does:
    from bitpredict.X import Y

All it does is register aliases so "bitpredict.X" resolves
to the correct local module:

    bitpredict.common.*   →  bitpredict_common/
    bitpredict.backtest.* →  backtest/
"""

import sys
import importlib
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _alias(bitpredict_name: str, real_name: str) -> None:
    try:
        sys.modules[bitpredict_name] = importlib.import_module(real_name)
    except ModuleNotFoundError:
        pass


# bitpredict.common → bitpredict_common/
_alias("bitpredict.common", "common")
_alias("bitpredict.common.constants", "common.constants")
_alias("bitpredict.common.logging", "common.logging")
_alias("bitpredict.common.data_loader", "common.data_loader")
_alias("bitpredict.common.db", "common.db")
_alias("bitpredict.common.db.services", "common.db.services")
_alias("bitpredict.common.db.services.data", "common.db.services.data")
_alias("bitpredict.common.ta", "common.ta")
_alias("bitpredict.common.ta.indicators", "common.ta.indicators")
_alias("bitpredict.common.stats", "common.stats")

# bitpredict.backtest.* → backtest/
# NOTE: "bitpredict.backtest" itself is NOT aliased here because
# backtest/__init__.py is what triggered this shim — it is still
# mid-loading, so importing it again would cause a circular import.
# Only its submodules are aliased (they are not mid-loading).
_alias("bitpredict.backtest.vectorbt_pro", "backtest.vectorbt_pro")
_alias(
    "bitpredict.backtest.vectorbt_pro.vbt_backtest",
    "backtest.vectorbt_pro.vbt_backtest",
)
_alias("bitpredict.backtest.vectorbt_pro.utils", "backtest.vectorbt_pro.utils")
_alias("bitpredict.backtest.custom", "backtest.custom")
_alias("bitpredict.backtest.custom.custom_backtest", "backtest.custom.custom_backtest")
_alias(
    "bitpredict.backtest.custom.templates.sl_base", "backtest.custom.templates.sl_base"
)
_alias(
    "bitpredict.backtest.custom.templates.static_tp_sl",
    "backtest.custom.templates.static_tp_sl",
)
_alias(
    "bitpredict.backtest.custom.templates.trailing_tp_sl",
    "backtest.custom.templates.trailing_tp_sl",
)

_alias("bitpredict.data.meta.utils", "bitpredict_data.meta.utils")
