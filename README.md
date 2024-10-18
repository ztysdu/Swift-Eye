# Swift-Eye: Towards Anti-blink Pupil Tracking for Precise and Robust High-Frequency Near-Eye Movement Analysis with Event Cameras


https://github.com/ztysdu/Swift-Eye/assets/69748689/d0ec23c4-2f40-432f-aaf0-fbb3a173a626

This is the implementation code for Swift-Eye, which was built upon [MMRotate: A Rotated Object Detection Benchmark using PyTorch](https://arxiv.org/pdf/2204.13317.pdf).

## Setup
After cloning our repositories, you can configure the environment by following these steps:

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

pip install -U openmim

mim install mmcv-full

mim install mmdet<3.0.0

cd mmrotate

pip install -v -e .

To ensure the installation was successful, you can verify it by checking the output of pip list, where you should see something like:

mmrotate                0.3.4       path/to/mmrotate

## Data
A test dataset is available for download [here](https://drive.google.com/drive/folders/1YXePrgSWd677JOKhVu9X_PUzqwv4D_49?usp=sharing). After downloading, please unzip the folder and place it in the `Swift_Eye/mmrotate/train_swift_eye` directory. If you require additional data, consider checking [EV-Eye](https://github.com/Ningreka/EV-Eye) and utilizing the code from [timelens](https://github.com/ztysdu/timelens).
[train_backbone_and_neck](https://drive.google.com/file/d/1Qy5BtB00_kk5sIbW40pUwJJYHkKpy2Qy/view?usp=sharing)

[train_with_temporal_fusion_component](https://drive.google.com/file/d/1CPtfQgR8WVcSe48SR1E51TJmCYk3tHLo/view?usp=sharing)

[train_without_temporal_fusion_component](https://drive.google.com/file/d/1ukt9VRmd3VWJh9KBZjS9N8QJGVif5vdt/view?usp=sharing)

[train Occlusion-ratio estimator](https://drive.google.com/file/d/1YhrRPm6TQ7ZVv5PUGhzg0yJOkAgmXJ8q/view?usp=sharing)

[test_data](https://drive.google.com/file/d/1MuInyfeuse1zrHjGaKsow2Vvg8Hz8MNq/view?usp=sharing)



## Model Weights
You can access the model weights from [this link](https://drive.google.com/file/d/18T-Kr_bDskaaowGCmdRbB8Hovzn8TEKH/view?usp=sharing). After downloading, kindly place the weighst in the `Swift_Eye/mmrotate/train_swift_eye/swift_eye` directory.

## Execution
To generate results and the corresponding videos, execute `/Swift-Eye/Swift-Eye/test_interpolated.py`.

## Comparison of Eye Movement Trajectories at 5000 FPS and 25 FPS

https://github.com/user-attachments/assets/69c1bc11-5362-4486-a0b5-1ca6707d2711

