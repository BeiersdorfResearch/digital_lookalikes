from typing import Sequence
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from PIL import Image


def test_selfie_integrity(path: Path | str):
    path = Path(path)
    try:
        img = Image.open(path.as_posix())
        img.verify()
    except (IOError, SyntaxError) as e:
        raise


def test_selfies(paths: Sequence[Path | str]):
    l = len(paths)
    with tqdm(total=l, ncols=100) as pbar:
        with ThreadPoolExecutor(max_workers=1000) as executor:
            futures = [executor.submit(test_selfie_integrity, path) for path in paths]
            # sourcery skip: no-loop-in-tests
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                    pbar.update(1)
                except Exception as e:
                    print(f"{future} raised an exception {e}")
                    pbar.update(1)
                    continue


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    save_dir = Path(hydra.utils.to_absolute_path(cfg.selfie_data.save_dir))
    paths = list(save_dir.glob("*/*.jpg"))
    test_selfies(paths)


if __name__ == "__main__":
    main()
