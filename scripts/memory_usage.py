import os

import psutil
from matplotlib import pyplot as plt
from noob import Tube, SynchronousRunner


def main():
    process = psutil.Process(os.getpid())
    tube = Tube.from_specification("test-memory")
    runner = SynchronousRunner(tube=tube)
    gen = runner.iter()
    ram_use_frame = []
    i = 0
    while True:
        try:
            next(gen)
            ram_used = process.memory_info().rss / (1024 * 1024)  # in MB
            ram_use_frame.append(round(ram_used, 2))
            i += 1
            if i % 100 == 0:
                print(f"{i} frames processed")
        except RuntimeError as e:
            print(e)
            break
    fig, ax = plt.subplots(figsize=(40, 8))
    plt.plot(ram_use_frame)
    plt.xlabel("frame")
    plt.ylabel("memory used (MB)")
    plt.tight_layout()
    plt.savefig("ram_use.svg", format="svg")


if __name__ == "__main__":
    main()
