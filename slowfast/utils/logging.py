#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Logging."""

import builtins
import decimal
import logging
import os
import sys
import simplejson

import slowfast.utils.distributed as du


def _suppress_print():
    """
    Suppresses printing from the current process.
    """

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    builtins.print = print_pass


def setup_logging(cfg):
    """
    Sets up the logging for multiple processes. Only enable the logging for the
    master process, and suppress logging for the non-master processes.
    """
    # Set up logging format.
    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"

    if du.is_master_proc(cfg.NUM_GPUS):
        # Enable logging for the master process.
        logging.root.handlers = []
        
        path = cfg.OUTPUT_DIR

        # logging.basicConfig(
        #     level=logging.INFO, format=_FORMAT, stream=sys.stdout
        # )

        logging.basicConfig(level=logging.INFO,
                        format=_FORMAT,
                        handlers=[logging.FileHandler(os.path.join(path,"out.log")),
                                logging.StreamHandler(sys.stdout)])

        logging.info(du.get_rank())
        logging.info(du.get_world_size())
    else:
        # Suppress logging for non-master processes.
        _suppress_print()


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)


def log_json_stats(stats):
    """
    Logs json stats.
    Args:
        stats (dict): a dictionary of statistical information to log.
    """
    stats = {
        k: decimal.Decimal("{:.6f}".format(v)) if isinstance(v, float) else v
        for k, v in stats.items()
    }
    json_stats = simplejson.dumps(stats, sort_keys=True, use_decimal=True)
    logger = get_logger(__name__)
    logger.info("json_stats: {:s}".format(json_stats))
