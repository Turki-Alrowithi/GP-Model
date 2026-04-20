"""Detection rules — geofence, crowd, weapon, intruder."""

from gpmodel.rules.base import Cooldown, Rule, RulesEngine
from gpmodel.rules.crowd import CrowdRule
from gpmodel.rules.geofence import Geofence, GeofenceRule
from gpmodel.rules.intruder import IntruderRule
from gpmodel.rules.weapon import WeaponRule

__all__ = [
    "Cooldown",
    "CrowdRule",
    "Geofence",
    "GeofenceRule",
    "IntruderRule",
    "Rule",
    "RulesEngine",
    "WeaponRule",
]
