# Swift-Eye: Towards Anti-blink Pupil Tracking for Precise and Robust High-Frequency Near-Eye Movement Analysis with Event Cameras


https://github.com/ztysdu/Swift-Eye/assets/69748689/d0ec23c4-2f40-432f-aaf0-fbb3a173a626

This is the implementation code for Swift-Eye, which was built upon [MMRotate: A Rotated Object Detection Benchmark using PyTorch](https://arxiv.org/pdf/2204.13317.pdf).

## Setup
For a smooth setup, we kindly suggest referring to [mmrotate](https://github.com/open-mmlab/mmrotate) to install mmrotate and the `requirements.txt` file in our project to set up the environment.

## Data
A test dataset is available for download [here](https://drive.google.com/drive/folders/1YXePrgSWd677JOKhVu9X_PUzqwv4D_49?usp=sharing). After downloading, please unzip the folder and place it in the `Swift_Eye/mmrotate/train_swift_eye` directory. If you require additional data, consider checking [EV-Eye](https://github.com/Ningreka/EV-Eye) and utilizing the code from timelens in the other folder.

## Model Weights
You can access the model weights from [this link](https://drive.google.com/file/d/1MprhEY5HoQKO-ZuFl7q_JyCu5l4oU_Zx/view?usp=sharing). After downloading, kindly place the weighst in the `Swift_Eye/mmrotate/train_swift_eye/swift_eye` directory.

## Execution
To generate results and the corresponding videos, execute `/Swift-Eye/Swift-Eye/test_interpolated.py`.

## Comparison of Eye Movement Trajectories at 5000 FPS and 25 FPS

https://github.com/user-attachments/assets/69c1bc11-5362-4486-a0b5-1ca6707d2711

