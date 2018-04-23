import torch
import argparse
import codecs
import nmt
import json
from torch import cuda
import nmt.utils.misc_utils as utils
import progressbar
from queue import Empty
from multiprocessing import Process, Queue
from collections import defaultdict
import sys
class QueueItem(object):
    """
    Models items in a queue.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TranslationHelper(object):

    def __init__(self, model_opt, test_opt, fields,
                 num_worker=2,use_cuda=False):
        """
        Loads translation models.
        """
        self._num_processes = num_worker
        self.model_opt = model_opt
        self.test_opt = test_opt
        self.fields =  fields
        self.use_cuda = use_cuda
        self._retrieved_translations = defaultdict(dict)
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

    def _load_model(self):
        model = nmt.model_helper.create_base_model(self.model_opt,self.fields)

        model.load_checkpoint(self.test_opt.model)
        if self.use_cuda:
            model = model.cuda()    
        return model

    def _load_translator(self):
        """
        Loads models and returns them.
        """
        model = self._load_model()

        translator = nmt.Translator(model=model, 
                                    fields=self.fields,
                                    beam_size=self.test_opt.beam_size, 
                                    n_best=1,
                                    max_length=self.test_opt.decode_max_length,
                                    global_scorer=None,
                                    cuda=self.use_cuda)
        # build and return models
        return translator

    def _start_worker(self, process_id, device_id):
        """
        Function executed by each worker once started. Do not execute in
        the parent process.
        """
        # load theano functionality
        translator = self._load_translator()

        # listen to queue in while loop, translate items
        while True:
            input_item = self._input_queue.get()

            if input_item is None:
                break
            batch = input_item.batch
            idx = input_item.idx
            request_id = input_item.request_id

            output_item = self._translate(translator, batch)
            self._output_queue.put((request_id, idx, output_item))

        return
    def _translate(self, translator, batch):
        ret = translator.translate_batch(batch)
        return ret


    ### WRITING TO AND READING FROM QUEUES ###

    def _send_jobs(self, data_iter, request_id=0):
        """
        """
        for idx, batch in enumerate(data_iter):

            input_item = QueueItem(
                                   seq=batch,
                                   idx=idx,
                                   request_id=request_id)

            self._input_queue.put(input_item)
        return idx+1

    def _retrieve_jobs(self, num_samples, request_id=0, timeout=5):
        """
        """
        while len(self._retrieved_translations[request_id]) < num_samples:
            resp = None
            while resp is None:
                try:
                    resp = self._output_queue.get(True, timeout)
                # if queue is empty after 5s, check if processes are still alive
                except Empty:
                    for midx in range(self._num_processes):
                        if not self._processes[midx].is_alive() and self._processes[midx].exitcode != 0:
                            # kill all other processes and raise exception if one dies
                            self._input_queue.cancel_join_thread()
                            self._output_queue.cancel_join_thread()
                            for idx in range(self._num_processes):
                                self._processes[idx].terminate()
                            sys.exit(1)
            request_id, idx, output_item = resp
            self._retrieved_translations[request_id][idx] = output_item
            #print self._retrieved_translations

        for idx in range(num_samples):
            yield self._retrieved_translations[request_id][idx]

        # then remove all entries with this request ID from the dictionary
        del self._retrieved_translations[request_id]

    ### EXPOSED TRANSLATION FUNCTIONS ###

    def translate(self, data_iter):
        """
        Returns the translation of source_segments.
        """
        n_samples = self._send_jobs(data_iter)

        for i, ret in enumerate(self._retrieve_jobs(n_samples)):
            print(ret)

            


def indices_lookup(indices,fields):

    words = [fields['tgt'].vocab.itos[i] for i in indices]
    sent = ' '.join(words)
    sent = sent.replace(' </s>','')
    return sent


def batch_indices_lookup(batch_indices,fields):

    batch_sents = []
    for sent_indices in batch_indices:
        sent = indices_lookup(sent_indices,fields)
        batch_sents.append(sent)
    return batch_sents



def translate_file(translator, 
                   data_iter, 
                   test_out, fields, 
                   use_cuda, 
                   dump_beam=None):

    print('start translating ...')
    with codecs.open(test_out, 'wb', encoding='utf8') as tgt_file:
        bar = progressbar.ProgressBar()

        for batch in bar(data_iter):
            ret = translator.translate_batch(batch)
            batch_sents = batch_indices_lookup(ret['predictions'][0], fields)
            for sent in batch_sents:
                tgt_file.write(sent+'\n')

        if dump_beam:
            print('dump beam ....')
            beam_trace = translator.beam_accum
            with codecs.open(dump_beam,'w',encoding='utf8') as f:
                json.dump(beam_trace,f)


# def main():

#     parser = argparse.ArgumentParser()
#     parser.add_argument("-test_data", type=str)
#     parser.add_argument("-test_out", type=str)
#     parser.add_argument("-config", type=str)
#     parser.add_argument("-model", type=str)
#     parser.add_argument("-data", type=str)
#     parser.add_argument("-dump_beam", default="", type=str)
#     parser.add_argument('-gpuid', default=[], nargs='+', type=int)
#     args = parser.parse_args()
#     opt = utils.load_hparams(args.config)





#     use_cuda = False
#     if args.gpuid:
#         cuda.set_device(args.gpuid[0])
#         use_cuda = True
#     fields = nmt.IO.load_fields(
#                 torch.load(args.data + '.vocab.pkl'))

#     test_dataset = nmt.IO.InferDataset(
#         data_path=args.test_data,
#         fields=[('src', fields["src"])])

#     test_data_iter = nmt.IO.OrderedIterator(
#                 dataset=test_dataset, device=args.gpuid[0],
#                 batch_size=1, train=False, sort=False,
#                 sort_within_batch=True, shuffle=False)


#     model = nmt.model_helper.create_base_model(opt,fields)


#     print('Loading parameters ...')

#     model.load_checkpoint(args.model)
#     if use_cuda:
#         model = model.cuda()    

#     translator = nmt.Translator(model=model, 
#                                 fields=fields,
#                                 beam_size=opt.beam_size, 
#                                 n_best=1,
#                                 max_length=opt.decode_max_length,
#                                 global_scorer=None,
#                                 cuda=use_cuda,
#                                 beam_trace=True if args.dump_beam else False)

#     translate_file(translator, test_data_iter, args.test_out, fields, use_cuda, args.dump_beam)
    


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-test_data", type=str)
    parser.add_argument("-test_out", type=str)
    parser.add_argument("-config", type=str)
    parser.add_argument("-model", type=str)
    parser.add_argument("-data", type=str)
    parser.add_argument("-dump_beam", default="", type=str)
    parser.add_argument('-gpuid', default=[], nargs='+', type=int)
    args = parser.parse_args()
    opt = utils.load_hparams(args.config)





    use_cuda = False
    if args.gpuid:
        cuda.set_device(args.gpuid[0])
        use_cuda = True
    fields = nmt.IO.load_fields(
                torch.load(args.data + '.vocab.pkl'))

    test_dataset = nmt.IO.InferDataset(
        data_path=args.test_data,
        fields=[('src', fields["src"])])

    test_data_iter = nmt.IO.OrderedIterator(
                dataset=test_dataset, device=args.gpuid[0],
                batch_size=1, train=False, sort=False,
                sort_within_batch=True, shuffle=False)

    trans_helper = TranslationHelper(opt,args,fields)
    trans_helper.translate(test_data_iter)
    

if __name__ == '__main__':
    main()