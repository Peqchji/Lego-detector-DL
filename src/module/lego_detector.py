import queue
import threading
import logging
import time
from entity import InferenceResponse, Job
from .port import AbstractJobHandler, AbstractObjectDetection

class LegoDetector():
    def __init__(
        self,
        object_dectection: AbstractObjectDetection,
        job_handler: AbstractJobHandler,
    ):
        self.object_dectection = object_dectection
        self.job_handler = job_handler

    def run(self):
        try:
            halt_event = threading.Event()
            job_chan = queue.Queue()
            response_chan = queue.Queue()

            def join_worker():
                halt_event.set()  # send halt signal
            
                for worker in workers:
                    worker.join()

            self.job_handler.register_job_callback(
                lambda job: self.__job_handler_worker(job, job_chan)
            )
            self.job_handler.set_halt_event(halt_event)
            
            workers = self.__run_mq_worker(halt_event, job_chan, response_chan)

            # run object detection on main thread
            self.__inference_worker(job_chan, response_chan, halt_event)

        except Exception as err:
            join_worker()
            raise err
        except KeyboardInterrupt:
            join_worker()
        finally:
            logging.info('Exiting process.')

    def __job_handler_worker(
        self, 
        job: Job,
        job_chan: queue.Queue, 
    ):
        job_chan.put(job)

    def __inference_worker(
        self, 
        job_chan: queue.Queue, 
        response_chan: queue.Queue, 
        halt_event: threading.Event
    ):
        while (not halt_event.is_set() or not job_chan.empty()):
            try:
                job: Job = job_chan.get(timeout=1)
            except queue.Empty:
                continue 

            if (job is None):
                continue
            
            start = time.time()

            preprocess_img = self.object_dectection.preprocess(job.image)
            output = self.object_dectection.inference(preprocess_img)
            results = self.object_dectection.postprocess(output)

            response = InferenceResponse(
                uid=job.uid,
                results=results,
                delivery_tag=job.delivery_tag
            )

            response_chan.put(response)

            logging.info(f'Job <{job.uid}> infernece time {(time.time() - start)*1000} ms')

    def __mark_done_worker(
        self, 
        response_chan: queue.Queue, 
        halt_event: threading.Event
    ):
        # use retry_queue instead of response_chan for data ownership
        # safer for getting front of queue cause retry_queue.queue[0] is not thread safe
        # might racecondition when use response_chan.queue[0]
        retry_queue = queue.Queue()

        while (not halt_event.is_set() or not response_chan.empty()):
            try:
                response: InferenceResponse = response_chan.get(timeout=1)
                
                if (response):
                    retry_queue.put(response)

            except queue.Empty:
                pass 

            while (not retry_queue.empty()):
                res = retry_queue.queue[0]
                is_success = self.job_handler.mark_job_as_done(res)

                if (is_success):
                    logging.info(f"Marked done (UID: {response.uid}, Tag: {response.delivery_tag})")
                    retry_queue.get_nowait() 
                else:
                    time.sleep(3)

    def __run_mq_worker(self, halt_event, job_chan: threading.Event, response_chan: threading.Event):
        workers = []
        tasks = [
            {'worker': self.__mark_done_worker, 'args': (response_chan, )}
        ]

        for t in tasks:
            worker = threading.Thread(target=t.get('worker'), args=(*t.get('args'), halt_event))
            workers.append(worker)
            worker.start()

        return workers