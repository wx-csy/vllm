import time
import tqdm
from .engine import LLMEngineExt
import uuid
import sys


def main(model_path: str = './parts/model_config/llama-13b/', tp: int = 1, pp: int = 1,
         bsz: int = 8, seq_len: int = 256, warmup_iter: int = 100, test_iter: int = 100,
         csv: bool = False, out_file: str = None):
    if out_file is None:
        of = sys.stdout
    else:
        of = open(out_file, 'a')
    engine = LLMEngineExt(model_path, pp, tp)
    for _ in range(bsz):
        engine.add_request(uuid.uuid4(), seq_len)
    with engine.no_process_output():
        engine.step()  # prompt run
        for _ in tqdm.tqdm(range(warmup_iter)):
            engine.step()
        if engine.bsz < bsz:
            print(f'Error: batch size reduced from {bsz} to {engine.bsz}')
            return
        start = time.monotonic()
        for _ in tqdm.tqdm(range(test_iter)):
            engine.step()
        elapsed = time.monotonic() - start

    sys.stdout = of
    if csv:
        print(f'{model_path},{pp},{tp},{bsz},{seq_len},{test_iter},{elapsed},{bsz*test_iter/elapsed},{elapsed/test_iter}')
    else:
        print('model_path:', model_path)
        print('pp:', pp)
        print('tp:', tp)
        print('iterations:', test_iter)
        print('seq_len:', seq_len)
        print('batch size:', bsz)
        print('time elapsed:', elapsed)
        print('throughput:', bsz * test_iter / elapsed)
        print('latency:', elapsed / test_iter)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
