from django.urls import path
from .views import RunDocumentFlowView, ListSalesOrderView, FilterDocTypeInformationView

urlpatterns = [
    path("documentflow/run", RunDocumentFlowView.as_view(), name="documentflow-run"),
    path("documentflow/sales-order-list", ListSalesOrderView.as_view(), name="documentflow-sales-order"),
    path("documentflow/doc-types/filter", FilterDocTypeInformationView.asview(), name="documentflow-doc-types-filter")
]