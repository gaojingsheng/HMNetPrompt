# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

class Task:
    '''
        This class is the ensemble of two classes: BatchGen and Eval.
        The `setup_task` function defines tasks w.r.t the three components based
        on the `task_name`.
    '''

    def __init__(self, batch_gen, evaluator):
        self.batch_gen = batch_gen
        self.evaluator = evaluator

    @classmethod
    def setup_task(cls, task_name, opt, save_dir):

        
        if task_name == 'HMNet':
            from Utils.HMNet.InfinibatchLoader import HMNetBatchGen
            batch_gen = HMNetBatchGen
            from Evaluation.ROUGEEval import ROUGEEval
            evaluator = ROUGEEval(opt['datadir'], save_dir, opt)
        elif task_name == 'Topicseg_pretrain':
            from Utils.HMNet.InfinibatchLoader_pretrain_topicseg import HMNetBatchGen
            batch_gen = HMNetBatchGen
            from Evaluation.TopicsegEval import TopicsegEval
        elif task_name == 'Topicseg':
            from Utils.HMNet.InfinibatchLoader_topicseg import HMNetBatchGen
            batch_gen = HMNetBatchGen
            from Evaluation.TopicsegEval import TopicsegEval
            evaluator = TopicsegEval(opt['datadir'], save_dir, opt)
        elif task_name == 'HMNet_topicseg':
            from Utils.HMNet.InfinibatchLoader_topicseg import HMNetBatchGen
            batch_gen = HMNetBatchGen
            from Evaluation.ROUGEEval import ROUGEEval
            evaluator = ROUGEEval(opt['datadir'], save_dir, opt)
        elif task_name == 'HMNet_pretrain':
            from Utils.HMNet.InfinibatchLoader_pretrain import HMNetBatchGen
            batch_gen = HMNetBatchGen
            from Evaluation.ROUGEEval import ROUGEEval
            evaluator = ROUGEEval(opt['datadir'], save_dir, opt)
        elif task_name == 'HMNet_pretrain_topicseg':
            from Utils.HMNet.InfinibatchLoader_pretrain_topicseg import HMNetBatchGen
            batch_gen = HMNetBatchGen
            from Evaluation.ROUGEEval import ROUGEEval
            evaluator = ROUGEEval(opt['datadir'], save_dir, opt)
        elif task_name == 'HMNet_pretrain_textrank':
            from Utils.HMNet.InfinibatchLoader_pretrain_textrank import HMNetBatchGen
            batch_gen = HMNetBatchGen
            from Evaluation.ROUGEEval import ROUGEEval
            evaluator = ROUGEEval(opt['datadir'], save_dir, opt)
        elif task_name == 'HMNet_pretrain_rouge':
            from Utils.HMNet.InfinibatchLoader_pretrain_rouge import HMNetBatchGen
            batch_gen = HMNetBatchGen
            from Evaluation.ROUGEEval import ROUGEEval
            evaluator = ROUGEEval(opt['datadir'], save_dir, opt)
        elif task_name == 'HMNet_onemask_textrank':
            from Utils.HMNet.InfinibatchLoader_onemask_textrank import HMNetBatchGen
            batch_gen = HMNetBatchGen
            from Evaluation.ROUGEEval import ROUGEEval
            evaluator = ROUGEEval(opt['datadir'], save_dir, opt)
        elif task_name == 'HMNet_onemask_rouge':
            from Utils.HMNet.InfinibatchLoader_onemask_rouge import HMNetBatchGen
            batch_gen = HMNetBatchGen
            from Evaluation.ROUGEEval import ROUGEEval
            evaluator = ROUGEEval(opt['datadir'], save_dir, opt)
        elif task_name == 'HMNet_rouge':
            from Utils.HMNet.InfinibatchLoader_rouge import HMNetBatchGen
            batch_gen = HMNetBatchGen
            from Evaluation.ROUGEEval import ROUGEEval
            evaluator = ROUGEEval(opt['datadir'], save_dir, opt)
        elif task_name == 'HMNet_rouge_topicseg':
            from Utils.HMNet.InfinibatchLoader_rouge_topicseg import HMNetBatchGen
            batch_gen = HMNetBatchGen
            from Evaluation.ROUGEEval import ROUGEEval
            evaluator = ROUGEEval(opt['datadir'], save_dir, opt)
        elif task_name == 'HMNet_textrank_topicseg':
            from Utils.HMNet.InfinibatchLoader_textrank_topicseg import HMNetBatchGen
            batch_gen = HMNetBatchGen
            from Evaluation.ROUGEEval import ROUGEEval
            evaluator = ROUGEEval(opt['datadir'], save_dir, opt)
        elif task_name == 'HMNet_mmr_topicseg':
            from Utils.HMNet.InfinibatchLoader_mmr_topicseg import HMNetBatchGen
            batch_gen = HMNetBatchGen
            from Evaluation.ROUGEEval import ROUGEEval
            evaluator = ROUGEEval(opt['datadir'], save_dir, opt)
        else:
            assert False
            print("ERROR: Task {} not defined".format(task_name))

        return cls(batch_gen, evaluator)
