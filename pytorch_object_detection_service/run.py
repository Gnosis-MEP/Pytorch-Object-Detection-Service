#!/usr/bin/env python
from event_service_utils.streams.redis import RedisStreamFactory
from event_service_utils.img_serialization.redis import RedisImageCache
from pytorch_object_detection_service.service import PytorchObjectDetectionService

from pytorch_object_detection_service.conf import (
    REDIS_ADDRESS,
    REDIS_PORT,
    PUB_EVENT_LIST,
    SERVICE_STREAM_KEY,
    SERVICE_CMD_KEY_LIST,
    LOGGING_LEVEL,
    TRACER_REPORTING_HOST,
    TRACER_REPORTING_PORT,
    SERVICE_DETAILS,
    OBJ_MODEL_NAME,
    DETECTION_THRESHOLD,
)


def run_service():
    tracer_configs = {
        'reporting_host': TRACER_REPORTING_HOST,
        'reporting_port': TRACER_REPORTING_PORT,
    }
    redis_fs_cli_config = {
        'host': REDIS_ADDRESS,
        'port': REDIS_PORT,
        'db': 0,
    }

    model_configs = {
        'hot_start': True,
        'cpu_only': True,
        'model_name': OBJ_MODEL_NAME,
        'detection_threshold': DETECTION_THRESHOLD,
    }
    file_storage_cli = RedisImageCache()
    file_storage_cli.file_storage_cli_config = redis_fs_cli_config
    file_storage_cli.initialize_file_storage_client()

    stream_factory = RedisStreamFactory(host=REDIS_ADDRESS, port=REDIS_PORT)
    service = PytorchObjectDetectionService(
        service_stream_key=SERVICE_STREAM_KEY,
        service_cmd_key_list=SERVICE_CMD_KEY_LIST,
        pub_event_list=PUB_EVENT_LIST,
        service_details=SERVICE_DETAILS,
        model_configs=model_configs,
        file_storage_cli=file_storage_cli,
        stream_factory=stream_factory,
        logging_level=LOGGING_LEVEL,
        tracer_configs=tracer_configs
    )
    service.run()


def main():
    try:
        run_service()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
