from rest_framework import serializers

class RunDocumentFlowSerializer(serializers.Serializer):
    module = serializers.CharField(max_length=50)
    db_schema = serializers.CharField(max_length=20)
    sales_order_no = serializers.CharField(max_length=20)
    project_id = serializers.CharField(max_length=64)

    def validate_sales_order_no(self, value: str) -> str:
        if value.isdigit():
            return value.zfill(10)
        return value


class ListSalesOrderSerializer(serializers.Serializer):
    db_schema = serializers.CharField(max_length=20)
    filter_conditions = serializers.JSONField(required=False, default=dict)



class FilterDocTypeInformationSerializer(serializers.Serializer):
    db_schema = serializers.CharField(max_length=20)
    doc_type = serializers.CharField(max_length=64)