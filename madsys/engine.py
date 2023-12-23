from contextlib import contextmanager
from typing import List
import uuid
from vllm.core.scheduler import SchedulerOutputs
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.arg_utils import EngineArgs
import vllm.engine.ray_utils as ray_utils
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import SamplerOutput


class LLMEngineExt(LLMEngine):
    process_output: bool = True

    def __init__(self, model: str, pp: int = 1, tp: int = 1):
        args = EngineArgs(model=model, load_format='dummy',
                          pipeline_parallel_size=pp, tensor_parallel_size=tp, enforce_eager=True)
        configs = args.create_engine_configs()
        par_configs = configs[2]
        dist_init_method, pg = ray_utils.initialize_cluster(par_configs)
        super().__init__(*configs, dist_init_method, pg, True)

    def add_request(self, uuid: uuid.UUID, length: int):
        sampling_params = SamplingParams(max_tokens=512, ignore_eos=True)
        super().add_request(str(uuid), None, sampling_params, [1] * length)

    def _process_model_outputs(self, output: SamplerOutput, scheduler_outputs: SchedulerOutputs) -> List[RequestOutput]:
        if self.process_output:
            return super()._process_model_outputs(output, scheduler_outputs)

    @contextmanager
    def no_process_output(self):
        old = self.process_output
        self.process_output = False
        yield
        self.process_output = old
