# smc/__init__.py
from bitpredict.common.ta.smc.base import SMCBase
from bitpredict.common.ta.smc.plot import plot
# Create instance for convenience
smc = SMCBase()

__all__ = ['smc', 'plot' ]