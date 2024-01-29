# Swift-Eye: Towards Anti-blink Pupil Tracking for Precise and Robust High-Frequency Near-Eye Movement Analysis with Event Cameras

[![Fast forward video](http://img.youtube.com/vi/0iu4A_2kjuk/0.jpg)](https://www.youtube.com/watch?v=0iu4A_2kjuk "YouTube video player")
This is the implementation code for Swift-Eye, which was built upon [MMRotate: A Rotated Object Detection Benchmark using PyTorch](https://arxiv.org/pdf/2204.13317.pdf).

## Setup
For a smooth setup, we kindly suggest referring to [mmrotate](https://github.com/open-mmlab/mmrotate) to install mmrotate and the `requirements.txt` file in our project to set up the environment.

## Data
A test dataset is available for download [here](https://drive.google.com/drive/folders/1YXePrgSWd677JOKhVu9X_PUzqwv4D_49?usp=sharing). After downloading, please unzip the folder and place it in the `Swift_Eye/mmrotate/train_swift_eye` directory. If you require additional data, consider checking [EV-Eye](https://github.com/Ningreka/EV-Eye) and utilizing the code from timelens in the other folder.

## Model Weights
You can access the model weights from [this link](https://drive.google.com/file/d/1MprhEY5HoQKO-ZuFl7q_JyCu5l4oU_Zx/view?usp=sharing). After downloading, kindly place the weighst in the `Swift_Eye/mmrotate/train_swift_eye/swift_eye` directory.

## Execution
To generate results and the corresponding videos, execute `/Swift-Eye/Swift-Eye/test_interpolated.py`.


