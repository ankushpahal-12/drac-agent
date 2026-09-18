"""
Chaos Fault Injection Package.
"""
from injector.fault_types import FAULT_SPECS
from injector.proxy import RuntimeFaultProxy

__all__ = ["FAULT_SPECS", "RuntimeFaultProxy"]
