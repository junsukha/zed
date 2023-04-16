///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2022, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////


#include <sl/Camera.hpp>
#include <fstream>
#include <sstream>
using namespace std;
using namespace sl;

int main(int argc, char **argv) {

    // Create a ZED camera object
    Camera zed;

    // Set configuration parameters
    InitParameters init_parameters;
    init_parameters.camera_resolution = RESOLUTION::HD1080; // Use HD1080 video mode
    init_parameters.camera_fps = 30; // Set fps at 30

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        cout << "Error " << returned_state << ", exit program." << endl;
        return EXIT_FAILURE;
    }

    // Capture 50 frames and stop
    int i = 0;
    sl::Mat image;
    //Mat.updateCPUfromGPU;

    while (i < 1) {
        // Grab an image
        returned_state = zed.grab();
        // A new image is available if grab() returns ERROR_CODE::SUCCESS
        if (returned_state == ERROR_CODE::SUCCESS) {

            // Get the left image
            zed.retrieveImage(image, VIEW::LEFT);

            // Get the right image
            zed.retrieveImage(image, VIEW::RIGHT);
            
            sl::uchar4 leftCenter;
            image.getValue<sl::uchar4>(image.getWidth() / 2, image.getHeight() / 2, &leftCenter);

            // Display the image resolution and its acquisition timestamp
            //cout<<"Image resolution: "<< image.getWidth()<<"x"<<image.getHeight() <<" || Image timestamp: "<<image.timestamp.data_ns<<endl;

            // display 
            // std::cout << "left image color B:" << (int)leftCenter[0] << std::endl;
            i++;
            cout << i << endl;
            const String filepath = "C:/Users/junsu/Documents/Brown/2023Spring/BVC/zed-examples-master/zed-examples-master/tutorials/tutorial-2-image-capture/cpp/example.txt";
            auto state = image.write(filepath);
            if (state == ERROR_CODE::SUCCESS) {
                cout << "check" << endl;
                cout << "Writing Successed.." << endl;
            }
            else {
                cout << "Writing failed.." << endl;
            }
        }
    }

    // Get camera information (ZED serial number)
    CalibrationParameters calibration_params = zed.getCameraInformation().camera_configuration.calibration_parameters;
    Transform t = calibration_params.stereo_transform;
    CameraParameters left_intrinsic_params = calibration_params.left_cam;
    CameraParameters right_intrinsic_params = calibration_params.right_cam;

    auto camera_infos = zed.getCameraInformation();
    auto cx = camera_infos.camera_configuration.calibration_parameters.left_cam.cx;
    auto cy = camera_infos.camera_configuration.calibration_parameters.left_cam.cy;

    // center
    auto left_fx = camera_infos.camera_configuration.calibration_parameters.left_cam.fx;
    auto left_fy = camera_infos.camera_configuration.calibration_parameters.left_cam.fy;

    // focal length
    auto right_fx = camera_infos.camera_configuration.calibration_parameters.right_cam.fx;
    auto right_fy = camera_infos.camera_configuration.calibration_parameters.right_cam.fy;


    auto rotMatrix = camera_infos.camera_configuration.calibration_parameters.stereo_transform.getRotationMatrix();
    auto traMatrix = camera_infos.camera_configuration.calibration_parameters.stereo_transform.getTranslation();
    
    ofstream myfile;
    myfile.open("example.txt");
    //myfile << image[0];

    ostringstream oss;
    /*oss << "example" << to_string(i) << ".txt";
    String filepath = oss.str();*/
    //const String filepath = "example" + to_string(i) + ".txt";
    
    /*for (auto i : image[0])
    cout << rotMatrix << endl;*/

    // Close the camera
    zed.close();
    return EXIT_SUCCESS;
}
