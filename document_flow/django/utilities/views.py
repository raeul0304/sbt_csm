from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.response import Response
import logging
from .serializers import RunDocumentFlowSerializer, ListSalesOrderSerializer, FilterDocTypeInformationSerializer
from .document_flow import run_documentflow_pipeline, get_sales_order_list_with_filter, get_doc_detail_info

logger = logging.getLogger(__name__)

# Document Flow 결과 조회
class RunDocumentFlowView(APIView):
    def post(self, request):
        serializer = RunDocumentFlowSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        module = serializer.validated_data["module"]
        db_schema = serializer.validated_data["db_schema"]
        sales_no = serializer.validated_data["sales_order_no"]
        project_id = serializer.validated_data["project_id"]

        try:
            result = run_documentflow_pipeline(db_schema, sales_no)

            if not result:
                return Response(
                    {"status": "result not found", "sales_order_no": sales_no},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            return Response(
                {
                    "status": "ok",
                    "module": module,
                    "project_id": project_id,
                    "sales_order_no": sales_no,
                    "doc_flow": result
                },
                status=status.HTTP_200_OK
            )
        
        except Exception as e:
            logger.exception("Document Flow Pipeline Failed")
            return Response(
                {
                    "status": "error",
                    "sales_order_no": sales_no,
                    "detail": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# Sales Order List 조회
class ListSalesOrderView(APIView):
    def post(self, request):
        serializer = ListSalesOrderSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        db_schema = serializer.validated_data['db_schema']
        filter_conditions = serializer.validated_data['filter_conditions']

        try:
            result = get_sales_order_list_with_filter(db_schema, filter_conditions)
            if not result:
                return Response(
                    {"status": "result not found", "filter_conditions": filter_conditions},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            return Response(
                {
                    "status": "ok",
                    "sales_order_list": result
                },
                status=status.HTTP_200_OK
            )
        
        except Exception as e:
            logger.exception("Document Flow Pipeline Failed")
            return Response(
                {
                    "status": "error",
                    "filter_conditions": filter_conditions,
                    "detail": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )



class FilterDocTypeInformationView(APIView):
    def post(self, request):
        serializer = FilterDocTypeInformationSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        db_schema = serializer.validated_data['db_schema']
        doc_type = serializer.validated_data['doc_type']

        try:
            result = get_doc_detail_info(db_schema, doc_type)

            if not result:
                return Response(
                    {
                        "status": "result not found", "doc_type": doc_type
                    },
                    status=status.HTTP_404_NOT_FOUND
                )
            
            return Response(
                {
                    "status" : "ok",
                    "doc_information": result
                },
                status = status.HTTP_200_OK
            )
        
        except Exception as e:
            logger.exception("Document Detail Information Not Found")
            return Response(
                {
                    "status": "error",
                    "doc_type": doc_type,
                    "detail": str(e)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )