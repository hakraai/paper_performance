"""Top-level convenience exports for the model_chain_inference package."""

from .plot import *  # noqa: F403
from .pymc_support import *  # noqa: F403
from .generate_data import *  # noqa: F403
from .bernstein import *  # noqa: F403
from .model_core import *  # noqa: F403
from .statistics import *  # noqa: F403
from .model_pymc import *  # noqa: F403
from .model_forecast import *  # noqa: F403
from .catalogue import *  # noqa: F403
from .data_prep import *  # noqa: F403
from .performance import *  # noqa: F403
from .testsuite import *  # noqa: F403


def _is_public_export(name, value):
	if name.startswith("_"):
		return False
	if name.isupper():
		return True
	module_name = getattr(value, "__module__", "")
	return module_name.startswith(__name__)


__all__ = [
	name for name, value in globals().items() if _is_public_export(name, value)
]

del _is_public_export
