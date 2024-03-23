import time
import logging
import torch
from pytorch_lightning.callbacks.callback import Callback
from lightning_utilities.core.rank_zero import rank_zero_only
from typing import Any, Dict
from megatron.core import parallel_state

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class NCCLInitializerCallback(Callback):

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logger.info("NCCLInitializerCallback run dummy_allreduce() and dummy_send_recv()")
        self.dummy_allreduce()
        self.dummy_send_recv()

    def dummy_allreduce(self):
        # Call the all_reduce collective with empty tensor to initialize the NCCL comm across all processes in parallel
        shape = (1, 1, 2)  # dummy shape
        tensor = torch.empty(shape, device=torch.cuda.current_device(), dtype=torch.float32)
        torch.distributed.all_reduce(tensor, group=parallel_state.get_tensor_model_parallel_group())

    def dummy_send_recv(self):
        # Let all participants in a pipeline to call send/recv at the same time to create the NCCL comm
        shape = (1, 2048, 1024)
        p2p_group = parallel_state.get_pipeline_model_parallel_group()
        ops = []
        tensor_send_prev = torch.empty(
            shape,
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=torch.float32
        )

        send_prev_op = torch.distributed.P2POp(
            op=torch.distributed.isend,
            tensor=tensor_send_prev,
            peer=parallel_state.get_pipeline_model_parallel_prev_rank(),
            group=p2p_group,
        )
        ops.append(send_prev_op)

        tensor_recv_prev = torch.empty(
            shape,
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=torch.float32
        )

        recv_prev_op = torch.distributed.P2POp(
            op=torch.distributed.irecv,
            tensor=tensor_recv_prev,
            peer=parallel_state.get_pipeline_model_parallel_prev_rank(),
            group=p2p_group,
        )
        ops.append(recv_prev_op)

        tensor_send_next = torch.empty(
            shape,
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=torch.float32
        )

        send_next_op = torch.distributed.P2POp(
            op=torch.distributed.isend,
            tensor=tensor_send_next,
            peer=parallel_state.get_pipeline_model_parallel_next_rank(),
            group=p2p_group,
        )
        ops.append(send_next_op)

        tensor_recv_next = torch.empty(
            shape,
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=torch.float32
        )

        recv_next_op = torch.distributed.P2POp(
            op=torch.distributed.irecv,
            tensor=tensor_recv_next,
            peer=parallel_state.get_pipeline_model_parallel_next_rank(),
            group=p2p_group,
        )
        ops.append(recv_next_op)

        reqs = torch.distributed.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
