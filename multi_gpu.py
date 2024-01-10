# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os

import torch

LOGGER = logging.getLogger(__name__)


def multi_gpu_barrier(rank):
    torch.distributed.barrier(device_ids=[rank])


def multi_gpu_receive_returns_from_rank_pathrenderer(rank, world_size, counter, returns):

    some_returns = returns.get_returns()

    returns.activate_mode(counter)

    async_ops = []
    new_tensors = []
    for _name, tensor in some_returns.items():
        this_tensor = torch.empty_like(tensor, device=rank)

        async_op = torch.distributed.irecv(this_tensor, src=counter % world_size)

        async_ops.append(async_op)
        new_tensors.append(this_tensor)

    for async_op in async_ops:
        async_op.wait()

    for name, tensor in zip(some_returns.keys(), new_tensors):
        returns.add_return(name, tensor.cpu())


def multi_gpu_send_returns_to_rank_pathrenderer(target_rank, returns):

    async_ops = []
    for tensor in returns.get_returns().values():
        async_op = torch.distributed.isend(tensor.cuda(), dst=target_rank)
        async_ops.append(async_op)

    for async_op in async_ops:
        async_op.wait()


def multi_gpu_sync_gradients(parameters):
    async_ops = []
    for param in parameters:
        param = param["parameters"]
        if param.grad is None:
            continue
        async_op = torch.distributed.all_reduce(
            param.grad, torch.distributed.ReduceOp.SUM, async_op=True
        )
        async_ops.append(async_op)
    for async_op in async_ops:
        async_op.wait()


def multi_gpu_setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)  # "29500"
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    # testing. also avoids a potential issue when barrier() is the first distributed function called.
    a = torch.ones(2) + rank
    a = a.to(rank)
    torch.distributed.all_reduce(a, op=torch.distributed.ReduceOp.SUM)
    b = a == 0  # force to actually sync a
    LOGGER.debug("testing parallelization on rank " + str(rank) + ": " + str(a) + " " + str(b))


def multi_gpu_cleanup(rank):
    multi_gpu_barrier(rank)
    torch.distributed.destroy_process_group()


def exception_logging_wrapper(rank, process_function, *args, **kwargs):
    try:
        process_function(rank, *args, **kwargs)
    except Exception as e:
        LOGGER.exception("EXCEPTION in rank " + str(rank) + ": " + str(e))
        raise


def multi_gpu_train(settings):

    world_size = torch.cuda.device_count()

    LOGGER.info("found " + str(world_size) + " GPUs")

    port = 29500
    import random

    port += random.randint(-100, +100)

    from train import train

    process_context = torch.multiprocessing.spawn(
        exception_logging_wrapper,
        args=(
            train,
            settings,
            world_size,
            port,
        ),
        nprocs=world_size,
        join=False,
    )

    try:
        process_context.join()
    except KeyboardInterrupt:
        LOGGER.warning("SHUTTING DOWN! DO NOT INTERRUPT AGAIN!")
        for i, process in enumerate(process_context.processes):
            if process.is_alive():
                LOGGER.info("terminating process " + str(i) + "...")
                process.terminate()
            process.join()
            LOGGER.info("process " + str(i) + " finished")
        raise
