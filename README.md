# Human Detection and Tracking Project

![Human Detection and Tracking](data/image.png)

## Overview

This personal project focuses on human detection and tracking using computer vision techniques. The primary objective is to identify and track human subjects in video streams or sequences of images, with potential applications in surveillance, robotics, and more. The project utilizes popular deep learning models and libraries to achieve precise human detection and tracking.

## Features

- **Human Detection:** Identifies and localizes humans within images or video frames.
- **Human Tracking:** Tracks detected humans across multiple frames to maintain their identity.
- **Real-time Processing:** Designed for real-time operation with live video streams.
- **Modular Design:** Easily extendable with different detection and tracking algorithms.
- **Visualization:** Provides visual feedback with bounding boxes around detected humans and tracking paths.

## Installation

1. **Clone the repository:**

```shell
 git clone https://github.com/vankhoa21991/human-dettrack.git
 cd human_dettrack
```

4. **Create a virtual environment (optional but recommended):**

```shell
python -m venv venv
source venv/bin/activate 
```

3. **Install dependencies:**

```shell
python -m venv venv
pip install -r requirements.txt
```

3. View Output: The application will display the video stream with bounding boxes around detected humans and tracking paths (if enabled).

## Getting Started

Please see [dataset.md](docs/dataset.md) and [quick_run.md](docs/quick_run.md) for the basic usage of MMTracking.

[//]: # (## Performance)

[//]: # (<div align="center">)

[//]: # ()
[//]: # (|  Tracker | HOTA↑ | MOTA↑ | IDF1↑ |)

[//]: # (| -------- | ----- | ----- | ----- |)

[//]: # (| [BoTSORT]&#40;https://arxiv.org/pdf/2206.14651.pdf&#41;    | 77.8 | 78.9 | 88.9 |)

[//]: # (| [DeepOCSORT]&#40;https://arxiv.org/pdf/2302.11813.pdf&#41; | 77.4 | 78.4 | 89.0 |)

[//]: # (| [OCSORT]&#40;https://arxiv.org/pdf/2203.14360.pdf&#41;     | 77.4 | 78.4 | 89.0 |)

[//]: # (| [HybridSORT]&#40;https://arxiv.org/pdf/2308.00783.pdf&#41; | 77.3 | 77.9 | 88.8 |)

[//]: # (| [ByteTrack]&#40;https://arxiv.org/pdf/2110.06864.pdf&#41;  | 75.6 | 74.6 | 86.0 |)

[//]: # (| [StrongSORT]&#40;https://arxiv.org/pdf/2202.13514.pdf&#41; |      | | |)

[//]: # (| <img width=200/>                                   | <img width=100/> | <img width=100/> | <img width=100/> |)

[//]: # ()
[//]: # (<sub> NOTES: performed on the 10 first frames of each MOT17 sequence. The detector used is ByteTrack's YoloXm, trained on: CrowdHuman, MOT17, Cityperson and ETHZ. Each tracker is configured with its original parameters found in their respective official repository.</sub>)

[//]: # ()
[//]: # (</div>)

## Contributing

Contributions are welcome! Feel free to fork the repository, make improvements, and submit pull requests. Please follow the Contributing Guidelines.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

If you have any questions or suggestions, please feel free to contact me:

Email: vankhoa21991@gmail.com
GitHub: github.com/vankhoa21991
Thank you for your interest in this project! Happy coding! 🚀
