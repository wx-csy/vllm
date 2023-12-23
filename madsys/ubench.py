import tqdm
from .engine import LLMEngineExt
import uuid


def main(model_path: str = './parts/model_config/llama-13b/', tp: int = 1, pp: int = 1):
    engine = LLMEngineExt(model_path, pp, tp)
    print('engine initialized')
    engine.add_request(uuid.uuid4(), 128)
    with engine.no_process_output():
        for _ in tqdm.tqdm(range(100)):
            engine.step()


if __name__ == '__main__':
    import fire
    fire.Fire(main)
