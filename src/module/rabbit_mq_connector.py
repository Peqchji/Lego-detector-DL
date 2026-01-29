import functools
import logging
import threading
import time
from typing import List, Tuple
import pika
from pika.channel import Channel
import pika.exceptions

from common.constant import QueueName
from common.utils import cal_backoff
from module.port import AbstractMQConnector

class RabbitMQConnector(AbstractMQConnector):
    _instance = None
    _lock = threading.Lock()
    _reconnect_thread = None


    __connection: pika.SelectConnection = None
    __channel: List[Channel] = [ None, None ]
    __consume_ready_event = threading.Event()
    __produce_ready_event = threading.Event()

    def __new__(cls, host, port, user, pwd):
        if (cls._instance is not None):
            logging.info('Has already connect to redis stack')

            return cls._instance

        cls._instance = super().__new__(cls)
        cls._instance.__start_ioloop(host, port, user, pwd)

        while(
            not cls._instance.__consume_ready_event.wait(0.5) and
            not cls._instance.__produce_ready_event.wait(0.5)
        ): pass

        return cls._instance
    
    def get_channel(self) -> Tuple[Channel, Channel]:
        return tuple(self.__channel)
    
    def get_connection(self) -> pika.SelectConnection:
        return self.__connection
    
    def close(self):
        try:
            for chann in self.__channel:
                if chann and chann.is_open:
                    chann.close()

            if self.__connection:
                try:
                    self.__connection.ioloop.stop()
                except Exception as e:
                    logging.warning(f"⚠️ IOLoop stop error: {e}")
                    
                if self.__connection.is_open:
                    self.__connection.close()

            logging.info("RabbitMQ connection closed successfully.")

        except pika.exceptions.ConnectionWrongStateError:
            logging.warning("Connection already closed. No action needed.")
        
        except Exception as e:
            logging.error(f"Error while closing RabbitMQ connection: {e}")

    def __start_ioloop(self, host, port, user, pwd) -> pika.SelectConnection:
        with self._lock:
            if self._reconnect_thread is None or not self._reconnect_thread.is_alive():
                self._reconnect_thread = threading.Thread(
                    target=self.__connect, args=(host, port, user, pwd), daemon=True
                )

                self._reconnect_thread.start()

    def __connect(self, host, port, user, pwd):
        reconnect_attempts = 1

        while (True):
            try:
                reconnector = functools.partial(
                        self.__reconnect, 
                        host, port, user, pwd,
                )
                
                logging.info(f"Connecting to RabbitMQ (Attempt {reconnect_attempts})...")
                
                self.__connection = pika.SelectConnection(
                    pika.ConnectionParameters(
                        host,
                        port,
                        credentials=pika.PlainCredentials(user, pwd),
                    ),
                    on_open_callback=self.__on_connection_open,
                    on_close_callback=reconnector,
                    on_open_error_callback=reconnector
                )

                logging.info("RabbitMQ connection established. Starting event loop...")
                self.__connection.ioloop.start()
                break  # Exit loop when connected successfully

            except pika.exceptions.AMQPError as e:
                reconnect_attempts += 1
                wait_time = cal_backoff(reconnect_attempts)

                logging.error(f"RabbitMQ connection failed. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

    def __on_connection_open(cls, connection: pika.SelectConnection):
        cls._instance.__connection = connection

        cls._instance.__connection.channel(
            on_open_callback=cls._instance.__on_consume_channel_open
        )

        cls._instance.__connection.channel(
            on_open_callback=cls._instance.__on_produce_channel_open
        )

    def __on_consume_channel_open(cls, channel: Channel):
        cls._instance.__channel[0] = channel

        channel.queue_declare(
            queue=QueueName.INFERENCE_SESSION, 
            durable=True, 
        )

        cls.__consume_ready_event.set()

    def __on_produce_channel_open(cls, channel: Channel):
        cls._instance.__channel[1] = channel

        channel.queue_declare(
            queue=QueueName.INFERENCE_RESPONSE, 
            durable=True, 
        )

        channel.queue_declare(
            queue=QueueName.INFERENCE_RESPONSE, 
            durable=True, 
        )

        cls.__produce_ready_event.set()

    def __reconnect(
        cls, 
        host, 
        port, 
        user, 
        pwd, 
        connection: pika.SelectConnection, 
        reason
    ):
        logging.warning(f"RabbitMQ connection lost: {reason}. Reconnecting...")
        cls._instance.__close()  # Close the current connection properly
        cls._instance.__start_ioloop()