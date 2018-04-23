from Queue import Empty
from multiprocessing import Process, Queue

class QueueItem(object):
    """
    Models items in a queue.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TranslationHelper(object):

    def __init__(self, model, fields,
                 beam_size, n_best=1,
                 max_length=100,
                 num_worker=2,
                 global_scorer=None, cuda=False,
                 beam_trace=False, min_length=0):
        """
        Loads translation models.
        """
        self._translators = settings.models
        self._num_processes = num_worker

        # set up queues
        self._init_queues()
        # init worker processes
        self._init_processes()

    def _init_queues(self):
        """
        Sets up shared queues for inter-process communication.
        """
        self._input_queue = Queue()
        self._output_queue = Queue()

    def shutdown(self):
        """
        Executed from parent process to terminate workers,
        method: "poison pill".
        """
        for process in self._processes:
            self._input_queue.put(None)

    def _init_processes(self):
        """
        Starts child (worker) processes.
        """
        processes = [None] * self._num_processes
        for process_id in range(self._num_processes):
            processes[process_id] = Process(
                target=self._start_worker,
                args=(process_id)
                )
            processes[process_id].start()

        self._processes = processes



    def _start_worker(self, process_id, device_id):
        """
        Function executed by each worker once started. Do not execute in
        the parent process.
        """
        # load theano functionality
        trng, fs_init, fs_next, gen_sample = self._load_models(process_id, device_id)

        # listen to queue in while loop, translate items
        while True:
            input_item = self._input_queue.get()

            if input_item is None:
                break
            idx = input_item.idx
            request_id = input_item.request_id

            output_item = self._translate(process_id, input_item, trng, fs_init, fs_next, gen_sample)
            self._output_queue.put((request_id, idx, output_item))

        return