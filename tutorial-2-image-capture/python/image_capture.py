########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

import pyzed.sl as sl
import cv2

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.camera_fps = 30  # Set fps at 30

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Capture 50 frames and stop
    i = 0
    image_l = sl.Mat(zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4)
    image_r = sl.Mat(zed.get_camera_information().camera_resolution.width, zed.get_camera_information().camera_resolution.height, sl.MAT_TYPE.U8_C4)
    runtime_parameters = sl.RuntimeParameters()
    while i < 10:
        # Grab an image, a RuntimeParameters object must be given to grab()
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # A new image is available if grab() returns SUCCESS
            zed.retrieve_image(image_l, sl.VIEW.LEFT)
            zed.retrieve_image(image_r, sl.VIEW.RIGHT)
            # timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # Get the timestamp at the time the image was captured
            # print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(image.get_width(), image.get_height(),
                #   timestamp.get_milliseconds()))


            image_zed_l = image_l.get_data()
            image_zed_r = image_r.get_data()

            # save image in current directory
            cv2.imwrite('left{}.jpg'.format(i), image_zed_l)
            cv2.imwrite('right{}.jpg'.format(i), image_zed_r)

            # show image in window
            # cv2.imshow('example', image_zed_l)
            # cv2.imshow('example2', image_zed_r)
            # cv2.waitKey(0)

            calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
            # Focal length of the left eye in pixels
            focal_left_x = calibration_params.left_cam.fx
            # First radial distortion coefficient
            k1 = calibration_params.left_cam.disto[0]
            # Translation between left and right sensors
            t = calibration_params.T
            # Horizontal field of view of the left eye in degrees
            h_fov = calibration_params.left_cam.h_fov



            i = i + 1
           
    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()
