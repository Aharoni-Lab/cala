import os
from cala.config import Config


def main():
    os.environ["video_directory"] = "1,2,3"
    print(Config().model_dump())


if __name__ == "__main__":
    main()
