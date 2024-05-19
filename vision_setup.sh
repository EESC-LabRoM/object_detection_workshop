#!/bin/bash

replace_config_param() {
    local config_file=$1
    local param_name=$2
    local new_value=$3

    sed -i "s|^#\($param_name=\)|\1|" $config_file
    sed -i "s|^\($param_name=\).*|\1$new_value|" $config_file
}

comment_config_param() {
    local config_file=$1
    local param_name=$2

    # Comment the line if it's not already commented
    sed -i "/^\($param_name=\)/s/^/#/" $config_file
}

echo "Please enter the camera's parameters:"

echo "Camera's mounting point [default=/dev/video0]"
read mount_point

if [[ ! $mount_point =~ ^/dev/.* ]]; then
    echo "Invalid input. Setting mounting point to default value of /dev/video0."
    mount_point="/dev/video0"
fi

echo "Camera width [default=1600]:"
read width

if [[ ! $width =~ ^[0-9]+$ ]]; then
    echo "Invalid input. Setting width to default value of 1600."
    width=1600
fi

echo "Camera height [default=1200]:"
read height

if [[ ! $height =~ ^[0-9]+$ ]]; then
    echo "Invalid input. Setting height to default value of 1200."
    height=1200
fi

echo "Camera FPS [default=10]:"
read fps

if [[ ! $fps =~ ^[0-9]+$ ]]; then
    echo "Invalid input. Setting FPS to default value of 10."
    fps=10
fi

main_dir="$(pwd)/../"
config_file="$(pwd)/vision_config.txt"

echo "Do you want to load the model in ONNX format? (y/n)"
read load_onnx

if [[ $load_onnx == "y" ]]; then
    echo "Please choose an ONNX model file from the following list:"
    onnx_dir=""$main_dir"onnx_models/"
    cd $onnx_dir

    select onnx_file in *.onnx; do
        if [ -n "$onnx_file" ]; then
            echo "Warning: Model calibration can take some time."
            break
        else
            echo "Invalid selection. Please try again."
        fi
    done
    full_model_path="$onnx_dir$onnx_file"
    
    replace_config_param $config_file "onnx-file" $full_model_path
    comment_config_param $config_file "model-engine-file"
else
    echo "Please choose a model engine file from the following list:"
    engine_dir=""$main_dir"engines/"
    cd $engine_dir

    select engine_file in *.engine; do
        if [ -n "$engine_file" ]; then
            break
        else
            echo "Invalid selection. Please try again."
        fi
    done
    full_engine_path="$engine_dir$engine_file"
    
    replace_config_param $config_file "model-engine-file" $full_engine_path
    comment_config_param $config_file "onnx-file"
fi

echo "Please enter the value of the confidence threshold [default=0.7]:"
read pre_cluster_threshold

if [[ ! $pre_cluster_threshold =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "Invalid input. Setting 'pre-cluster-threshold' to default value of 0.7."
    pre_cluster_threshold=0.7
fi

replace_config_param $config_file "pre-cluster-threshold" $pre_cluster_threshold
replace_config_param $config_file "labelfile-path" ""$main_dir"labels.txt"
replace_config_param $config_file "custom-lib-path" ""$main_dir"libnvds_infercustomparser_tlt.so"

v4l2_resolution_command="v4l2-ctl --set-fmt-video=width=$width,height=$height,pixelformat=1"
v4l2_fps_command="v4l2-ctl --set-fmt-video=width=$width,height=$height,pixelformat=1"
echo "Running v4l2 commands"

eval $v4l2_resolution_command
eval $v4l2_fps_command

echo "Configuration and v4l2 setup updated successfully."

export PYTHONPATH=$PYTHONPATH:"$main_dir/bindings/build"
export PGIE_CONFIG=$config_file
export CAMERA_WIDTH=$width
export CAMERA_HEIGHT=$height
export CAMERA_FPS=$fps

echo "Do you want to run the Python vision script? (y/n)"
read run_python_script

if [[ $run_python_script == "y" ]]; then
    export PYTHONPATH=$PYTHONPATH:"$main_dir/bindings/build"
    echo "Running Python vision script..."
    python3 ""$main_dir"vision/vision.py" $mount_point
    echo "Skipping Python vision script."
fi
