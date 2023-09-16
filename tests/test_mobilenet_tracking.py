# author: Jorge Chong
# author: Marlon Zambrano

from pathlib import Path
import numpy as np
import os
import sys
import cv2
import depthai as dai
from config import CONFIG
import blobconverter
import time

class VeloPipeline():
    """Setup Pipeline for Velocam"""
    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Dictionary with the configuration
        """
        self._config = config
        self._pipeline = None
    
    def setup(self):
        """Creates the pipeline using the configuration provided"""
        self._pipeline = dai.Pipeline()
        
        # Mono Left
        mono_l = self._pipeline.create(dai.node.MonoCamera)
        # Mono Right
        mono_r = self._pipeline.create(dai.node.MonoCamera)
        # Stereo Depth
        stereo = self._pipeline.create(dai.node.StereoDepth)
        # Tracker Node
        tracker = self._pipeline.create(dai.node.ObjectTracker)

        # RGB Camera
        cam_rgb = self._pipeline.create(dai.node.ColorCamera)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setPreviewSize(
            self._config["RGB_WIDTH"], 
            self._config["RGB_HEIGHT"]
        )
        cam_rgb.setInterleaved(False)

        # Left and Right Mono
        mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_l.setCamera("left")
        mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_r.setCamera("right")

        # Align depth map to the perspective of RGB camera
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(mono_l.getResolutionWidth(), mono_l.getResolutionHeight())

        # Spatial Detection Network
        detection_nn = self._pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        detection_nn.setBlobPath(
            blobconverter.from_zoo(name='vehicle-detection-adas-0002', shaves=5)
        )
        detection_nn.setConfidenceThreshold(self._config["DETECTION_CONFIDENCE"])
        detection_nn.input.setBlocking(False)
        detection_nn.setBoundingBoxScaleFactor(0.5)
        detection_nn.setDepthLowerThreshold(100)
        detection_nn.setDepthUpperThreshold(5000)

        # mono_l -> stereo
        mono_l.out.link(stereo.left)
        # mono_r -> stereo
        mono_r.out.link(stereo.right)
        # color camera -> spatial detection
        cam_rgb.preview.link(detection_nn.input)
        
        # If full frame mode active:
        if self._config["FULL_FRAME_TRACKING"]:
            cam_rgb.setPreviewKeepAspectRatio(False)
            cam_rgb.video.link(tracker.inputTrackerFrame)
            tracker.inputTrackerFrame.setQueueSize(2)
        else:
            detection_nn.passthrough.link(tracker.inputTrackerFrame)

        detection_nn.passthrough.link(tracker.inputDetectionFrame)
        detection_nn.out.link(tracker.inputDetections)
        stereo.depth.link(detection_nn.inputDepth)

        # RGB output
        xout_rgb = self._pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("preview")
        # cam_rgb.preview.link(xout_rgb.input)
        # Inference output
        # xout_nn = self._pipeline.create(dai.node.XLinkOut)
        # xout_nn.setStreamName("nn")
        # detection_nn.out.link(xout_nn.input)

        # Trackler output
        tracker_out = self._pipeline.create(dai.node.XLinkOut)
        tracker_out.setStreamName("tracklets")

        # tracker -> xout_rgb
        tracker.passthroughTrackerFrame.link(xout_rgb.input)
        tracker.out.link(tracker_out.input)

        return self._pipeline

def normalize_frame(frame, bbox):
    norms = np.full(len(bbox), frame.shape[0])
    norms[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norms).astype(int)



if __name__ == "__main__":
    pipeline = VeloPipeline(CONFIG).setup()
    print(CONFIG)
    print(pipeline)
    TRACKED_OBJECTS = {}
    LABEL_MAP = ["", "auto"]

    with dai.Device(pipeline=pipeline, usb2Mode=CONFIG['DEVICE']['USB2MODE']) as device:

        print("Connected Cameras: ", device.getConnectedCameras())
        # Usb Speed
        print("USB speed: ", device.getUsbSpeed().name)
        # Bootloader version
        if device.getBootloaderVersion() is not None:
            print("Bootloader Version: ", device.getBootloaderVersion())

        q_preview = device.getOutputQueue("preview")
        q_tracklets = device.getOutputQueue("tracklets")

        frame = None
        detections = []

        start_time = time.monotonic()
        counter = 0
        fps = 0

        while True:
            counter += 1
            current_time = time.monotonic()
            if (current_time - start_time) > 1:
                fps = counter / (current_time - start_time)
                counter = 0
                start_time = current_time


            in_rgb = q_preview.tryGet()
            in_tracklets = q_tracklets.tryGet()

            if in_rgb is not None and in_tracklets is not None:
                frame = in_rgb.getCvFrame()
                tracklets_data = in_tracklets.tracklets
                for tr in tracklets_data:
                    roi = tr.roi.denormalize(frame.shape[1], frame.shape[0])
                    x1 = int(roi.topLeft().x)
                    y1 = int(roi.topLeft().y)
                    x2 = int(roi.bottomRight().x)
                    y2 = int(roi.bottomRight().y)

                    # Update tracked list of objects for trail
                    if tr.status.name == 'NEW':
                        TRACKED_OBJECTS[tr.id] = []
                        TRACKED_OBJECTS[tr.id].append(
                            (x1,y1,x2,y2,tr.label,tr.status.name,tr.spatialCoordinates.x,tr.spatialCoordinates.y,tr.spatialCoordinates.z)
                        )
                    if tr.status.name == 'LOST':
                        try:
                            ltr = len(TRACKED_OBJECTS[tr.id])
                            if ltr > 1:
                                TRACKED_OBJECTS[tr.id] = TRACKED_OBJECTS[tr.id][1:]
                            else:
                                TRACKED_OBJECTS[tr.id] = []
                        except KeyError:
                            pass
                    if tr.status.name == 'TRACKED':
                        try:
                            TRACKED_OBJECTS[tr.id].append(
                                (x1,y1,x2,y2,tr.label,tr.status.name,tr.spatialCoordinates.x,tr.spatialCoordinates.y,tr.spatialCoordinates.z)
                            )
                            ltr = len(TRACKED_OBJECTS[tr.id])
                            if ltr > CONFIG["TRACK_LIMIT"]:
                                TRACKED_OBJECTS[tr.id] = TRACKED_OBJECTS[tr.id][1:]
                        except KeyError:
                            pass
                    if tr.status.name == 'REMOVED':
                        try:
                            del TRACKED_OBJECTS[tr.id]
                        except KeyError:
                            pass


                    try:
                        label = LABEL_MAP[tr.label]
                    except:
                        label = tr.label

                    cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, 255)
                    cv2.putText(frame, f"ID: {[tr.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_DUPLEX, 0.5, 255)
                    cv2.putText(frame, tr.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), cv2.FONT_HERSHEY_DUPLEX)

                    cv2.putText(frame, f"X: {int(tr.spatialCoordinates.x)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_DUPLEX, 0.5, 255)
                    cv2.putText(frame, f"Y: {int(tr.spatialCoordinates.y)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_DUPLEX, 0.5, 255)
                    cv2.putText(frame, f"Z: {int(tr.spatialCoordinates.z)} mm", (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_DUPLEX, 0.5, 255)

            #if frame is not None:
            #    for detection in detections:
            #        # bbox normalization
            #        bbox = normalize_frame(frame, 
            #            (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
            #        )
            #        # confidence
            #        conf = detection.confidence
            #        # class label
            #        label = detection.label
            #        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

                # Show fps
                cv2.putText(frame, "FPS: {:.2f}".format(fps), (2, frame.shape[0]-4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
                cv2.imshow("preview", frame)

            # Close
            if cv2.waitKey(1) == ord('q'):
                break