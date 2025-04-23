# yolov5-cpp

Object Detection using YOLOv5 and C++

![Image Detection](./sample_detected.jpg)

![Video Detection](./sample_detected.gif)

## YOLOv5 Model conversion

```sh
# Clone YOLOv5 repository
git clone https://github.com/ultralytics/yolov5.git

cd yolov5

# Download the model weight file, here take yolov5s.pt as example
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt

# Convert yolov5s.pt to yolov5s.onxx
python export.py --weights yolov5s.pt --include onnx
```

## 🖥️ Getting Started

1. Clone the Repository

```sh
git clone https://github.com/jiafie7/yolov5-cpp.git
cd yolov5-cpp
```

2. Execution

```sh
# Build project
mkdir build && cd build
cmake ..
make

# Single image mode
./main --image ../sample.jpg

# Batch process mode
./main --batch ../data ../results

# Video process mode
./main --video ../src.mp4 ../dest.mp4

# Camera real-time mode
./main --camera
./main --camera ../dest.mp4
```

## License

This project is licensed under the MIT License. See the LICENSE file for more information.

## Contributing

Contributions, bug reports, and feature requests are welcome. Feel free to fork the repository, open issues, or submit pull requests.
