from dataclasses import dataclass
from typing import Optional


@dataclass
class ProductFeatures:
    """
    Normalized representation of rule-relevant product characteristics.

    Optional booleans are used to distinguish:
    - explicit absence of a feature
    - feature not yet determined
    """

    has_worst_of: Optional[bool] = None
    has_autocall: Optional[bool] = None
    underlying_type: Optional[str] = None
    basket_size: Optional[int] = None