import json
import logging
import threading

from collections.abc import Callable
from typing import Tuple

import pika
from common.constant import QueueName
from common.utils import cal_backoff
from entity import Job, InferenceResponse
from .rabbit_mq_connector import RabbitMQConnector
from .port import AbstractJobHandler
from pika.channel import Channel


class RabbitMQJobHandler(AbstractJobHandler):
    def __init__(
            self, 
            mq_connector: RabbitMQConnector,
        ):
        self.__mq_connector = mq_connector
        self.__get_job_callback = None

        self.__produce_channel: Channel = None
        self.__consume_channel: Channel = None

        self.__is_callback_ready = threading.Event()
        self.__halt_event: threading.Event = threading.Event()

        threading.Thread(
            target=self.__channel_health_ckeck
        ).start()

    def set_halt_event(self, event: threading.Event):
        self.__halt_event = event

    def register_job_callback(self, callback: Callable[[Job], None]):
        self.__get_job_callback = callback
        self.__is_callback_ready.set()

    def mark_job_as_done(self, job_result: InferenceResponse) -> bool:
        success_flag = False
        response = json.dumps({
                'uid': job_result.uid,
                'status': job_result.status,
                'results': job_result.results
            })
        
        try:
            if self.__produce_channel and not self.__produce_channel.is_closed:
                self.__produce_channel.basic_publish(
                    exchange="",
                    routing_key=QueueName.INFERENCE_RESPONSE,
                    body=response,
                    properties=pika.BasicProperties(delivery_mode=2),
                )

                self.__consume_channel.basic_ack(job_result.delivery_tag)
                success_flag = True
            else:
                logging.error("Channel is closed. Cannot publish message.")
        except Exception as err:
            logging.error(f"Failed to mark job as done for Job._id: {job_result.uid}, Error: {err}")

        # self.__connection.ioloop.call_later(0, publish_callback)
        # success_flag.wait()
        
        return success_flag
    
    def __setup_channel(self):
        self.__consume_channel, self.__produce_channel = self.__mq_connector.get_channel()

        if self.__consume_channel.is_closed or self.__produce_channel.is_closed:
            return

        self.__consume_channel.queue_declare(queue=QueueName.INFERENCE_SESSION, durable=True)
        self.__consume_channel.queue_declare(queue=QueueName.INFERENCE_RESPONSE, durable=True)
        
        self.__consume_channel.basic_consume(
            queue=QueueName.INFERENCE_SESSION,
            on_message_callback=self.__get_job,
            auto_ack=False,
        )
        
        logging.info("Job Getter is available!")

    def __channel_health_ckeck(self):
        attempts = 0
        min_wait_time = 20.0
        wait_time = min_wait_time

        is_channel_down = lambda: (
            not self.__produce_channel or self.__produce_channel.is_closed or
            not self.__consume_channel or self.__consume_channel.is_closed
        )

        while not self.__halt_event.is_set():
            if (is_channel_down()):
                wait_time = cal_backoff(attempts, min_wait_time, 30)
                self.__setup_channel()

                if (not is_channel_down()):
                    continue

                logging.error(f"Channel is not available!. retry in {wait_time}")

                attempts += 1
            else:
                attempts = 0
                wait_time = min_wait_time
                logging.info(f"Channel is available!.")

            threading.Event().wait(wait_time)

    def __get_job(self, ch: Channel, method, properties, body) -> None:
        try:
            # declare if need
            self.__consume_channel.queue_declare(queue=QueueName.INFERENCE_SESSION, durable=True)

            val: dict = json.loads(body)
            uid = val.get('uid')
            image = val.get('image').encode('latin-1')
            job = Job(uid, image, method.delivery_tag)

            logging.info(f"Recieving job (UID: {uid}, Tag: {method.delivery_tag})")

            self.__is_callback_ready.wait()

            if (self.__get_job_callback):
                self.__get_job_callback(job)

        except Exception as err:
            logging.error(f"Error processing job: {err}")
            ch.basic_nack(method.delivery_tag, requeue=True)