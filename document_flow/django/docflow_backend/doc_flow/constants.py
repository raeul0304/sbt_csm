# myapp/constants.py
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Tuple

class Group(str, Enum):
    SALES_ORDER = "SALES_ORDER"
    DELIVERY = "DELIVERY"
    BILLING = "BILLING"
    ACCOUNTING = "ACCOUNTING"

# 그룹
GROUP_LABELS: Mapping[Group, Tuple[str, ...]] = MappingProxyType({
    Group.SALES_ORDER: ("Sales Order",),
    Group.DELIVERY: ("Outbound Delivery", "Picking Request", "GD Goods Issue", "RE Goods Delivery"),
    Group.BILLING: ("Invoice", "Cancel Invoice"),
    Group.ACCOUNTING: ("Journal Entry",),
})

# 레이아웃 상수
X_BY_GROUP: Mapping[Group, int] = MappingProxyType({
    Group.SALES_ORDER: 0,
    Group.DELIVERY: 150,
    Group.BILLING: 300,
    Group.ACCOUNTING: 450,
})

DEFAULT_GROUP: Group = Group.DELIVERY
Y_GAP: int = 150

# 흐름 규칙
RULES: Mapping[str, Tuple[str, ...]] = MappingProxyType({
    "J": ("C",),      # Outbound Delivery <- Sales Order
    "Q": ("J",),      # Picking Request <- Outbound Delivery
    "R": ("Q", "h"),  # GD Goods Issue <- Picking Request
    "M": ("R",),      # Invoice <- GD Goods Issue
    "h": ("R",),      # RE Goods Delivery <- GD Goods Issue
    "N": ("M",),      # Cancel Invoice <- Invoice
    "E": ("M", "N"),  # Journal Entry <- Invoice or Cancel Invoice
})

# 역매핑 : doc -> group
DOC_TYPE_TO_GROUP: Mapping[str, Group] = MappingProxyType({
    label.strip().lower(): group
    for group, labels in GROUP_LABELS.items()
    for label in labels
})