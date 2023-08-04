import multiprocessing
from .trinitygenerateFORTEST import generationLoop
import torch

queue = multiprocessing.Queue()


def init(number_of_gpu: int):
    torch.multiprocessing.set_start_method("spawn")
    processes = list()
    for i in range(number_of_gpu):
        device_id = f"cuda:{i}"
        p = multiprocessing.Process(
            target=generationLoop, args=(queue, device_id), daemon=True
        )
        processes.append(p)
    for thread in processes:
        thread.start()


def getGeneration(text: str, generation_type: int):
    r"""generation type : 1 => 도서 추천사유 시작부 생성
    generation type : 2 => 도서별 추천사유"""
    outQueue = multiprocessing.Queue()
    queue.put((text, generation_type, outQueue))
    outQueue.get()
