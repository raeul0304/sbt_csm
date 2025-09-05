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


#문서 매핑
DOCUMENT_MAP = {
    'Sales Order': {
        'doc_no': 'vbak.vbeln',
        'item': 'vbap.posnr',
        'preced_doc': 'vbap.vgbel',
        'orig_item': 'vbap.vgpos',
        'quantity': 'vbap.kwmeng',
        'unit': 'vbap.vrkme',
        'ref_value': 'vbap.netwr',
        'curr': 'vbak.waerk',
        'created_on': 'vbak.erdat',
        'material': 'vbap.matnr',
        'description': 'vbap.arktx',
        'status': 'vbak.gbstk'
    },
    'Outbound Delivery': {
        'doc_no': 'likp.vbeln',
        'item': 'lips.posnr',
        'preced_doc': 'lips.vgbel',
        'orig_item': 'lips.vgpos',
        'quantity': 'lips.lfimg',
        'unit': 'lips.vrkme',
        'created_on': 'likp.erdat',
        'material': 'lips.matnr',
        'description': 'lips.arktx',
        'status': 'likp.gbstk'
    },
    'Picking Request': {
        'doc_no': 'likp.erdat',
        'item': 'lips.posnr',
        'preced_doc': 'likp.vbeln',
        'orig_item': 'lips.vgpos',
        'quantity': 'lips.lfimg',
        'unit': 'lips.vrkme',
        'created_on': 'likp.erdat',
        'material': 'lips.matnr',
        'description': 'lips.arktx',
        'status': 'likp.gbstk'
    },
    'GD Goods Issue': {
        'doc_no': 'mseg.mblnr',
        'item': 'mseg.zeile',
        'preced_doc': 'mseg.vbeln_im',
        'orig_item': 'mseg.vbelp_im',
        'quantity': 'mseg.menge',
        'unit': 'mseg.meins',
        'ref_value': 'mseg.dmbtr',
        'curr': 'mseg.waers',
        'created_on': 'mseg.cpudt_mkpf',
        'material': 'mseg.matnr',
        'description': 'lips.arktx',
        'status': 'likp.wbstk'
    },
    'RE Goods Delivery': {
        'doc_no': 'mseg.mblnr',
        'item': 'mseg.zeile',
        'preced_doc': 'mseg.vbeln_im',
        'orig_item': 'mseg.vbelp_im',
        'quantity': 'mseg.menge',
        'unit': 'mseg.meins',
        'ref_value': 'mseg.dmbtr',
        'curr': 'mseg.waers',
        'created_on': 'mseg.cpudt_mkpf',
        'material': 'mseg.matnr',
        'description': 'lips.arktx',
        'status': 'likp.gbstk'
    },
    'Invoice': {
        'doc_no': 'vbrk.vbeln',
        'item': 'vbrp.posnr',
        'preced_doc': 'vbrp.vgbel',
        'orig_item': 'vbrp.vgpos',
        'quantity': 'vbrp.fkimg',
        'unit': 'vbrp.vrkme',
        'ref_value': 'vbrp.netwr',
        'curr': 'vbrk.waerk',
        'created_on': 'vbrk.erdat',
        'material': 'vbrp.matnr',
        'description': 'vbrp.arktx',
        'status': 'vbrk.gbstk'
    },
    'Cancel Invoice': {
        'doc_no': 'vbrk.vbeln',
        'item': 'vbrp.posnr',
        'preced_doc': 'vbrk.stblg',
        'orig_item': 'vbrp.vgpos',
        'quantity': 'vbrp.fkimg',
        'unit': 'vbrp.vrkme',
        'ref_value': 'vbrp.netwr',
        'curr': 'vbrk.waerk',
        'created_on': 'vbrk.erdat',
        'material': 'vbrp.matnr',
        'description': 'vbrp.arktx',
        'status': 'vbrk.gbstk'
    },
    'Journal Entry': {
        'doc_no': 'bseg.belnr',
        'preced_doc': 'bseg.vbeln',
        'ref_value': 'bseg.wrbtr',
        'curr': 'bkpf.waers',
        'created_on': 'bkpf.cpudt',
        'status': 'bseg.augbl'
    }
}
