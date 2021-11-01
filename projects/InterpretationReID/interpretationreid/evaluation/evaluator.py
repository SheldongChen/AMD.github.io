# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import logging
import time
from contextlib import contextmanager

import torch
import copy
from fastreid.utils.logger import log_every_n_seconds


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.
    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.
    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def preprocess_inputs(self, inputs):
        pass

    def process(self, inputs, outputs):
        """
        Process an input/output pair.
        Args:
            inputs: the inputs that's used to call the model.
            outputs: the return value of `model(input)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.
        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:
                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass




def inference_on_dataset(model, data_loader, evaluator,name_of_attribute=None):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.
    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.

        outputs_dict:
                {
                    "outputs": outputs,
                    "att_list": att_list,
                    "fake_attributes": fake_attributes,
                    "real_attributes": real_attributes
                }

    Returns:
        The return value of `evaluator.evaluate()`


    """
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader.dataset)))

    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0

    evaluator.name_of_attribute(name_of_attribute)


    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()



            outputs_dict = model(inputs)
            outputs = outputs_dict["outputs"]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            evaluator.process(inputs, outputs)
            evaluator.process_for_visualize(inputs, outputs_dict)

            idx += 1
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_batch = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_batch > 30:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / batch. ETA={}".format(
                        idx + 1, total, seconds_per_batch, str(eta)
                    ),
                    n=30,
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / batch per device)".format(
            total_time_str, total_time / (total - num_warmup)
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / batch per device)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup)
        )
    )
    results = evaluator.evaluate()

    #TODO CXD Visualizer
    dist_list_stack,query_real_attributes, gallery_real_attributes = evaluator.visualize() # n x m x NUM_ATT
    dict_q_att = {3:'Reweight with the most contributed attribute, unstable operation!', 4:"Reweight with the most contributed exclusive attribute, stable operation! "}
    for list_q_att in list([3,4]):
        logger.info("*" * 50)
        logger.info("Orginal Results: {}".format(results))
        logger.info("*" * 50)

        New_results_best = None

        for p_att in list(torch.arange(0.0,2.0,0.05)):

            #logger.info("*" * 50)
            for q_att in list([list_q_att]):
                New_results = evaluator.evaluate_dist(dist_list_stack,q_att,p_att,query_real_attributes)
                #logger.info(" mode={} , p_att={} ".format(q_att,p_att))
                #logger.info("{}".format(New_results))
                if New_results_best is None or New_results['New_Rank-1']>=New_results_best['New_Rank-1']:
                    New_results_best = copy.deepcopy(New_results)
            #logger.info("*" * 50)
        logger.info("*" * 50)
        logger.info("mode = {}, New Reweight Results = {}".format(dict_q_att[q_att],New_results_best))
        logger.info("*" * 50)







    logger.info("*" * 50)
    New_results = evaluator.evaluate_dist(dist_list_stack, q_att=None, p_att=None)
    logger.info("{}".format(New_results))
    logger.info("*" * 50)


    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    explain_result = evaluator.explain_eval(dist_list_stack, query_real_attributes, gallery_real_attributes, lamda=2.0)


    if results is None:
        results = {}
    else:
        results.update(**New_results)
        results.update(**explain_result)
        print(explain_result)
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.
    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
