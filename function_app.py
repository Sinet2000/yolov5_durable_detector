import json
import logging
import os

import azure.durable_functions as df
import azure.functions as func

from managers import AzureBlobManager, AzureTableStorageManager
from models import BlobToProcessQueueMessage, BlobProcessStatus, VisioDetectorHttpRequest, DetectorType, ImagePredictionResult
from detectors import run_yolov_detector
from visio_detector import VisioDetector

blob_container_name = os.environ.get("BlobContainerName")
blob_connection_string = os.environ.get("BlobConnectionString")
azure_blob_manager = AzureBlobManager(blob_connection_string, blob_container_name)

predictions_blob_container_name = os.environ.get("ProcessedBlobsContainerName")
predictions_azure_blob_manager = AzureBlobManager(blob_connection_string, predictions_blob_container_name)

table_storage_name = os.environ.get("TableStorageName")
table_connection_string = os.environ.get("TableConnectionString")
azure_table_storage_manager = AzureTableStorageManager(table_connection_string, table_storage_name)

def delete_file_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File '{file_path}' deleted successfully.")
    else:
        print(f"File '{file_path}' does not exist.")

# We can provide a key, and use function level: https://learn.microsoft.com/en-us/azure/azure-functions/functions-bindings-http-webhook-trigger?tabs=python-v2%2Cisolated-process%2Cnodejs-v4%2Cfunctionsv2&pivots=programming-language-python
app = df.DFApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.function_name(name="ObjectDetectionHttpTrigger")
@app.route(route="yolov/detect", methods=("POST",))
@app.durable_client_input(client_name="client")
async def http_start(req: func.HttpRequest, client: df.DurableOrchestrationClient):
    req_body = req.get_body().decode('utf-8')
    logging.info(f"Started ObjectDetectionHttpTrigger, received: {req_body}")

    instance_id = await client.start_new("image_detection_orchestrator", client_input=req_body)
    
    await client.wait_for_completion_or_create_check_status_response(req, instance_id)     

    # Get orchestration execution status
    status = await client.get_status(instance_id)     

    # Retrieves orchestration execution results and displays them on the screen
    runtime = status.runtime_status
    output = status.output
    logging.info(f"runtime: {runtime}\n\n output:{output}")

    # response = client.create_check_status_response(req, instance_id)
    # body_bytes = response.get_body()
    # body_str = body_bytes.decode('utf-8')  # Assuming UTF-8 encoding, adjust if necessary
    # logging.info(f"ObjectDetectionHttpTrigger.response body: {body_str}")
    # return response

    return output

@app.orchestration_trigger(context_name="context")
def image_detection_orchestrator(context: df.DurableOrchestrationContext):
    visio_detector_req_str = context.get_input()
    visio_detector_json = json.loads(visio_detector_req_str)
    logging.info(f"image_detection_orchestrator.visio_detector_json: {visio_detector_json}")

    try:
        visio_detector_req = VisioDetectorHttpRequest.from_json(visio_detector_json)
        logging.info(f"image_detection_orchestrator.detector_type: {visio_detector_req.detector_type}")

        if visio_detector_req.detector_type == DetectorType.YoloV5:
            result = yield context.call_activity("run_yolov_detection_activity", visio_detector_req_str)
        else:
            result = ImagePredictionResult(
                image_name=visio_detector_req.file_name,
                detector_type=visio_detector_req.detector_type,
                errors= "The detector type is incorrect",
                has_errors=True
                ).to_json()

        logging.info(f"image_detection_orchestrator.result: {result}")
        return result
    except Exception as ex:
        logging.error(f"An unexpected error occurred: {ex}")
        return ImagePredictionResult(
            image_name=visio_detector_json['fileName'],
            detector_type=DetectorType.UNKNOWN,
            errors= str(ex),
            has_errors=True
            ).to_json()


@app.activity_trigger(input_name="visioDetectorReqStr")
def run_yolov_detection_activity(visioDetectorReqStr: str) -> str:
    visioDetectorModel = VisioDetectorHttpRequest.from_json(json.loads(visioDetectorReqStr))

    logging.info(f"Running YOLOv5 detection for: {visioDetectorModel.file_name}")

    try:
        logging.info(f"Detector Type: {visioDetectorModel.detector_type}")

        file_download_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image_set/yolov5')
        downloaded_file_path = azure_blob_manager.download_and_upload_file(visioDetectorModel.file_name, file_download_dir)

        yolov5_precition_result = VisioDetector.run_yolov_detector_wrapper(visioDetectorModel.file_name, downloaded_file_path)

        predictions_azure_blob_manager.upload_file_to_blob(yolov5_precition_result.result_img_path, yolov5_precition_result.result_img_name)

        logging.info(f"run_yolov_detection_activity.Result: {yolov5_precition_result}")

        # delete_file_if_exists(downloaded_file_path)
        return yolov5_precition_result.to_json()

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return ImagePredictionResult(
            image_name=visioDetectorModel.file_name,
            detector_type=visioDetectorModel.detector_type,
            errors= str(e),
            has_errors=True
            ).to_json()

# @app.activity_trigger(input_name="visioDetectorReqStr")
# def run_ssd_detection_activity(visioDetectorReqStr: str) -> str:
#     logging.info(f"run_ssd_detection_activity: {type(visioDetectorReqStr)}")

#     visioDetectorModel = VisioDetectorHttpRequest.from_json(json.loads(visioDetectorReqStr))

#     logging.info(f"Running SSD detection for: {visioDetectorModel.file_name}")
    
#     try:
#         logging.info(f"Detector Type: {visioDetectorModel.detector_type}")
#         return ImagePredictionResult(
#             image_name=visioDetectorModel.file_name,
#             detector_type=visioDetectorModel.detector_type,
#             prediction= 0.90
#             ).to_json()
    
#         # azure_blob_manager.download_and_upload_file(blob_process_queue_message.file_name, source_img_dir)

#     except Exception as e:
#         logging.error(f"An unexpected error occurred: {e}")
#         return ImagePredictionResult(
#             image_name=visioDetectorModel.file_name,
#             detector_type=visioDetectorModel.detector_type,
#             errors= str(e),
#             has_errors=True
#             ).to_json()

# @app.queue_trigger(arg_name="azqueue", queue_name="blob-py-test-queue", connection="AzureWebJobsStorage")
# def queue_trigger(azqueue: func.QueueMessage):
#     queue_message = azqueue.get_body().decode('utf-8')

#     client = df.DurableOrchestrationClient(starter)
#     instance_id = yield client.start_new(orchestrator_function, None, azqueue)
#     return instance_id