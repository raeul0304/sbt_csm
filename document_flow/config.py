from enum import Enum
from types import MappingProxyType
from typing import Mapping, Tuple

DB_NAME = 'doc_flow_db'
DB_USER = 'llm_user'
DB_PASSWORD = '1234'
DB_HOST = '192.168.1.154'
DB_PORT = '8004'


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

#테이블 매핑
TABLE_MAP = {
    'Sales Order': {'table': 'vbak', 'doc_col': 'vbeln'},
    'Outbound Delivery': {'table': 'likp', 'doc_col': 'vbeln'},
    'Invoice': {'table': 'vbrk', 'doc_col': 'vbeln'},
    'Cancel Invoice': {'table': 'vbrk', 'doc_col': 'vbeln'},
    'GD Goods Issue': {'table': 'mseg', 'doc_col': 'mblnr'},
    'RE Goods Delivery': {'table': 'mseg', 'doc_col': 'mblnr'},
    'Journal Entry': {'table': 'bseg', 'doc_col': 'belnr'},
    'Picking Request': {'table': 'likp', 'doc_col': 'vbeln'},
}

#컬럼 매핑
COLUMN_MAP = {
    'Sales Order': {
        'doc_no': 'vbeln',
        'item': 'posnr',
        'preced_doc': 'vgbel',
        'orig_item': 'vgpos',
        'quantity': 'kwmeng',
        'unit': 'vrkme',
        'ref_value': 'netwr',
        'curr': 'waerk',
        'created_on': 'erdat',
        'material': 'matnr',
        'description': 'arktx',
        'status': 'gbstk'
    },
    'Outbound Delivery': {
        'doc_no': 'vbeln',
        'item': 'posnr',
        'preced_doc': 'vgbel',
        'orig_item': 'vgpos',
        'quantity': 'lfimg',
        'unit': 'vrkme',
        'created_on': 'erdat',
        'material': 'matnr',
        'description': 'arktx',
        'status': 'gbstk'
    },
    'Picking Request': {
        'doc_no': 'erdat',
        'item': 'posnr',
        'preced_doc': 'vbeln',
        'orig_item': 'vgpos',
        'quantity': 'lfimg',
        'unit': 'vrkme',
        'created_on': 'erdat',
        'material': 'matnr',
        'description': 'arktx',
        'status': 'gbstk'
    },
    'GD Goods Issue': {
        'doc_no': 'mblnr',
        'item': 'zeile',
        'preced_doc': 'vbeln_im',
        'orig_item': 'vbelp_im',
        'quantity': 'menge',
        'unit': 'meins',
        'ref_value': 'dmbtr',
        'curr': 'waers',
        'created_on': 'cpudt_mkpf',
        'material': 'matnr',
        'description': 'arktx',
        'status': 'wbstk'
    },
    'Invoice': {
        'doc_no': 'vbeln',
        'item': 'posnr',
        'preced_doc': 'vgbel',
        'orig_item': 'vgpos',
        'quantity': 'fkimg',
        'unit': 'vrkme',
        'ref_value': 'netwr',
        'curr': 'waerk',
        'created_on': 'erdat',
        'material': 'matnr',
        'description': 'arktx',
        'status': 'gbstk'
    },
    'Cancel Invoice': {
        'doc_no': 'vbeln',
        'item': 'posnr',
        'preced_doc': 'stblg',
        'orig_item': 'vgpos',
        'quantity': 'fkimg',
        'unit': 'vrkme',
        'ref_value': 'netwr',
        'curr': 'waerk',
        'created_on': 'erdat',
        'material': 'matnr',
        'description': 'arktx',
        'status': 'gbstk'
    },
    'Journal Entry': {
        'doc_no': 'belnr',
        'preced_doc': 'vbeln',
        'ref_value': 'wrbtr',
        'curr': 'waers',
        'created_on': 'cpudt',
        'status': 'augbl'
    }
}
