#!/usr/bin/env python3

"""
Object Detection Pipeline using GStreamer and DeepStream

This script sets up a GStreamer pipeline to process video input from a V4L2 device,
perform object detection using a deep learning model, and display the results with
overlays indicating the number and type of detected objects.

Functions:
    load_config_file(config_file_path): Load the configuration file and extract the label file path.
    load_label_file(label_file_path): Load labels from the label file and create a mapping of class names to IDs.
    process_frame(frame_meta, object_classes, batch_meta): Process each frame, count objects, and update display metadata.
    create_pipeline(device_path): Create and configure the GStreamer pipeline and its elements.
    link_elements(elements): Link the GStreamer elements within the pipeline.
    osd_sink_pad_buffer_probe(pad, info, u_data): Probe function to handle metadata extraction and display updates.
    main(args): Main function to parse arguments, set up the pipeline, and start the event loop.

"""

import os
import sys
import logging
from datetime import datetime

sys.path.append('../')

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import pandas as pd
import pyds
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call

# Global configuration variables
PGIE_FILE = os.environ.get('PGIE_CONFIG', './config_infer_primary_yoloV8.txt')
WIDTH = os.environ.get('WIDTH', 1920)
HEIGHT = os.environ.get('HEIGHT', 1080)
FPS = os.environ.get('FPS', 30)

def load_config_file(config_file_path):
    """
    Load the PGIE configuration file and extract the label file path.

    Args:
        config_file_path (str): Path to the PGIE configuration file.

    Returns:
        str: Path to the label file.

    Raises:
        ValueError: If the label file path is not found in the configuration file.
    """
    with open(config_file_path, 'r') as config_file:
        lines = config_file.readlines()
    for line in lines:
        if 'labelfile-path=' in line:
            return line.split('=')[1].strip()
    raise ValueError("Label file path not found in config file")

def load_label_file(label_file_path):
    """
    Load labels from the label file and create a dictionary mapping class names to IDs.

    Args:
        label_file_path (str): Path to the label file.

    Returns:
        dict: Dictionary mapping class names to their respective IDs.
    """
    with open(label_file_path, 'r') as label_file:
        lines = label_file.read().splitlines()
    return {line: i for i, line in enumerate(lines)}

def process_frame(frame_meta, object_classes, batch_meta):
    """
    Process each frame, count objects, and update display metadata.

    Args:
        frame_meta: Frame metadata.
        object_classes (dict): Dictionary mapping class names to IDs.
        batch_meta: Batch metadata.

    Returns:
        None
    """
    obj_counter = {id: 0 for _, id in object_classes.items()}
    frame_number = frame_meta.frame_num
    num_rects = frame_meta.num_obj_meta
    l_obj = frame_meta.obj_meta_list

    while l_obj is not None:
        try:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
        except StopIteration:
            break
        obj_counter[obj_meta.class_id] += 1
        try:
            l_obj = l_obj.next
        except StopIteration:
            break

    current_time = datetime.now().strftime("%H:%M:%S")
    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    display_meta.num_labels = 1
    py_nvosd_text_params = display_meta.text_params[0]
    display_txt = f"Frame Number={frame_number} Number of Objects={num_rects}"
    
    for k, v in obj_counter.items():
        _class = [cls for cls in object_classes.keys() if object_classes[cls] == k][0]
        display_txt += f" {_class}_count={v}"
    
    py_nvosd_text_params.display_text = display_txt
    py_nvosd_text_params.x_offset = 10
    py_nvosd_text_params.y_offset = 12
    py_nvosd_text_params.font_params.font_name = "Serif"
    py_nvosd_text_params.font_params.font_size = 10
    py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
    py_nvosd_text_params.set_bg_clr = 1
    py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)


def create_pipeline():
    """
    Create and configure the GStreamer pipeline and its elements.

    Returns:
        dict: Dictionary containing the created GStreamer elements.

    Raises:
        RuntimeError: If any GStreamer element cannot be created.
    """
    pipeline = Gst.Pipeline()
    if not pipeline:
        raise RuntimeError("Unable to create Pipeline")

    source = Gst.ElementFactory.make("v4l2src", "usb-cam-source")
    caps_v4l2src = Gst.ElementFactory.make("capsfilter", "v4l2src_caps")
    vidconvsrc = Gst.ElementFactory.make("videoconvert", "convertor_src1")
    nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "convertor_src2")
    caps_vidconvsrc = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    sink = Gst.ElementFactory.make("nv3dsink" if is_aarch64() else "nveglglessink", "nvvideo-renderer")

    elements = {
        "pipeline": pipeline,
        "source": source,
        "caps_v4l2src": caps_v4l2src,
        "vidconvsrc": vidconvsrc,
        "nvvidconvsrc": nvvidconvsrc,
        "caps_vidconvsrc": caps_vidconvsrc,
        "streammux": streammux,
        "pgie": pgie,
        "nvvidconv": nvvidconv,
        "nvosd": nvosd,
        "sink": sink
    }
    
    for name, element in elements.items():
        if not element:
            raise RuntimeError(f"Unable to create {name}")
    
    return elements

def link_elements(elements):
    """
    Link the GStreamer elements within the pipeline.

    Args:
        elements (dict): Dictionary containing the GStreamer elements.

    Returns:
        None

    Raises:
        RuntimeError: If any element cannot be linked.
    """
    elements["pipeline"].add(elements["source"])
    elements["pipeline"].add(elements["caps_v4l2src"])
    elements["pipeline"].add(elements["vidconvsrc"])
    elements["pipeline"].add(elements["nvvidconvsrc"])
    elements["pipeline"].add(elements["caps_vidconvsrc"])
    elements["pipeline"].add(elements["streammux"])
    elements["pipeline"].add(elements["pgie"])
    elements["pipeline"].add(elements["nvvidconv"])
    elements["pipeline"].add(elements["nvosd"])
    elements["pipeline"].add(elements["sink"])
    
    elements["source"].link(elements["caps_v4l2src"])
    elements["caps_v4l2src"].link(elements["vidconvsrc"])
    elements["vidconvsrc"].link(elements["nvvidconvsrc"])
    elements["nvvidconvsrc"].link(elements["caps_vidconvsrc"])
    
    sinkpad = elements["streammux"].get_request_pad("sink_0")
    srcpad = elements["caps_vidconvsrc"].get_static_pad("src")
    
    srcpad.link(sinkpad)
    elements["streammux"].link(elements["pgie"])
    elements["pgie"].link(elements["nvvidconv"])
    elements["nvvidconv"].link(elements["nvosd"])
    elements["nvosd"].link(elements["sink"])

def set_element_properties(elements, device_path):
    """
    Set properties on the GStreamer elements.

    Args:
        elements (dict): Dictionary containing the GStreamer elements.
        device_path (str): Path to the V4L2 device.

    Returns:
        None
    """
    elements["caps_v4l2src"].set_property('caps', Gst.Caps.from_string(f"video/x-raw, framerate={FPS}/1"))
    elements["caps_vidconvsrc"].set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM)"))
    elements["source"].set_property('device', device_path)
    elements["streammux"].set_property('width', WIDTH)
    elements["streammux"].set_property('height', HEIGHT)
    elements["streammux"].set_property('batch-size', 1)
    elements["streammux"].set_property('batched-push-timeout', 4000000)
    elements["pgie"].set_property('config-file-path', PGIE_FILE)
    elements["sink"].set_property('sync', False)


def osd_sink_pad_buffer_probe(pad, info, u_data):
    """
    Probe function to handle metadata extraction and display updates.

    Args:
        pad: The pad to which the probe is attached.
        info: Buffer information.
        u_data: User data (unused).

    Returns:
        Gst.PadProbeReturn: Status of the probe operation.
    """
    label_file_path = load_config_file(PGIE_FILE)
    object_classes = load_label_file(label_file_path)
    
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK
    
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        
        process_frame(frame_meta, object_classes, batch_meta)
        
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    
    return Gst.PadProbeReturn.OK

def main(args):
    """
    Main function to parse arguments, set up the pipeline, and start the event loop.

    Args:
        args (list): Command-line arguments.

    Returns:
        int: Exit status.
    """

    Gst.init(None)
    loop = GLib.MainLoop()
    
    device_path = "/dev/video0" if len(args) < 2 else args[1]
    elements = create_pipeline()
    link_elements(elements)
    set_element_properties(elements, device_path)
    
    bus = elements["pipeline"].get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)
    
    osdsinkpad = elements["nvosd"].get_static_pad("sink")
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    
    logger.info("Starting pipeline")
    elements["pipeline"].set_state(Gst.State.PLAYING)
    
    try:
        loop.run()
    except:
        pass
    finally:
        elements["pipeline"].set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
