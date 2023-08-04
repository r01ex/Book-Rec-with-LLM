import multiprocessing
from opensourceLLM.trinitygenerateFORTEST import generationLoop
import torch
import json
import time

queue = multiprocessing.Queue()
outqueue = multiprocessing.Queue()


def getGeneration(text: str, generation_type: int, thread: int):
    r"""generation type : 1 => 도서 추천사유 시작부 생성
    generation type : 2 => 도서별 추천사유"""
    queue.put((text, generation_type, thread))
    result, Outthread = outqueue.get()
    if Outthread == thread:
        return result
    else:
        outqueue.put((result, Outthread))
        time.sleep(0.5)


if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)
    import process

    process.app.run(host="0.0.0.0", port=80)
    if config["modelchoice"] == "opensourceLLM":
        torch.multiprocessing.set_start_method("spawn")
        processes = list()
        for i in range(config["Number_of_GPU_for_Generation"]):
            device_id = f"cuda:{i}"
            p = multiprocessing.Process(
                target=generationLoop, args=(queue, device_id, outqueue), daemon=True
            )
            processes.append(p)
        for generate_process in processes:
            generate_process.start()
