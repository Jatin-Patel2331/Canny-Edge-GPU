// Reference: Adapted and extended from the Canny Edge Detection implementation available at https://github.com/akwCode/CED/blob/main/CED.cpp
//
// Modifications: The original `main` function has been refactored into multiple functions for improved modularity and readability. The OpenCL kernel for Canny edge detection (`canny.cl`) was developed independently.

#include <stdio.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cstddef>
#include <vector>

// OpenCV header
#include <opencv2/opencv.hpp>

// Undefine Status if defined
#ifdef Status
#undef Status
#endif

// OpenGL and X11 headers
//#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/glx.h>

#include <CT/DataFiles.hpp>
#include <Core/Assert.hpp>
#include <Core/Image.hpp>
#include <Core/Time.hpp>
#include <OpenCL/Device.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/cl-patched.hpp>

#include <boost/lexical_cast.hpp>

using namespace std;
using namespace cv;
using namespace cl;

bool runCpu = true;
bool runGpu = true;
bool displayGpu = true;
bool writeImages = false;

string selectImage();

int cpuFunction ( const Mat& src );
Mat GaussianFilter ( const Mat& src );
Mat NonMaximumSuppression ( const Mat& magnitude, const Mat& blurred, const Mat& angle );
void DoubleThresholding ( const Mat& magnitude, Mat& strong, Mat& weak );
Mat Hysteresis ( Mat& strong, const Mat& weak );
void chooseStageToDisplay ( const Mat* GPtr, const Mat* BlPtr, const Mat* sblPtr, const Mat* NMSPtr, const Mat* DTSPtr, const Mat* DTWPtr, const Mat* FinalPtr );

int main ( int argc, char* argv[] ) {
    cout << "Beginning of the project!" << endl;

    // GPU setup ------------------------------------------
    // Create Context
    cl::Context context ( CL_DEVICE_TYPE_GPU );
    // Device list
    int deviceNr = argc < 2 ? 1 : atoi ( argv[1] );
    cout << "Using device " << deviceNr << " / "
         << context.getInfo<CL_CONTEXT_DEVICES>().size() << std::endl;
    ASSERT ( deviceNr > 0 );
    ASSERT ( ( std::size_t ) deviceNr <= context.getInfo<CL_CONTEXT_DEVICES>().size() );
    cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>() [deviceNr - 1];
    std::vector<cl::Device> devices;
    devices.push_back ( device );
    OpenCL::printDeviceInfo ( std::cout, device );

    // Create a command queue
    cl::CommandQueue queue ( context, device, CL_QUEUE_PROFILING_ENABLE );

    // Load the source code
    extern unsigned char canny_cl[];
    extern unsigned int canny_cl_len;
    cl::Program program ( context,
                          std::string ( ( const char* ) canny_cl,
                                        canny_cl_len ) );

    OpenCL::buildProgram ( program, devices );

    // ----------------------------------------------------

    // Allocate space for output data from CPU and GPU on the host
    // std::vector<float> h_input ( count ); // not used
    // std::vector<float> h_outputCpu ( count ); // not used

    // ----------------------------------------------------
    char decision;
    do {
        string imgName = selectImage();
        if ( imgName == "q" ) {
            cout << "Exiting the program..." << endl;
            return 0;
            }
        string imgPath = "../img/" + imgName;
        if ( imgPath.empty() ) {
            cout << "No image selected. Exiting program." << endl;
            continue;
            }

        Mat img = imread ( imgPath, IMREAD_GRAYSCALE );

        // for ( std::size_t j = 0; j < countY; j++ ) {
        //     for ( std::size_t i = 0; i < countX; i++ ) {
        //         h_input[i + countX * j] = imgVector[ ( i % cols ) + cols * ( j % rows )];
        //         }
        //     }
        // for ( size_t k = 0; k < h_input.size(); k++ )
        //     cout << h_input[k] << " ";


        Core::TimeSpan cpuStart = Core::getCurrentTime();
        cpuFunction ( img );
        Core::TimeSpan cpuEnd = Core::getCurrentTime();

        cout << "CPU implementation successfully!" << endl;

        // GPU ----------------------------------------------------
        if ( img.type() != CV_32F ) {
            img.convertTo ( img, CV_32F, 1/255.0 );
            }
        std::vector<float> imgVector;
        imgVector.assign ( ( float* ) img.datastart, ( float* ) img.dataend );

        // Get the maximum work-group size for the device using C++ API
        std::size_t maxWorkGroupSize;
        device.getInfo ( CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize );

        std::size_t baseWgSize = std::sqrt ( maxWorkGroupSize ); // Calculate a base work-group size
        std::size_t wgSizeX = std::min ( baseWgSize, static_cast<std::size_t> ( 16 ) );
        std::size_t wgSizeY = wgSizeX; // Same as wgSizeX for square work groups

        // Obtain image dimensions
        std::size_t imgWidth = img.cols;
        std::size_t imgHeight = img.rows;

        // Adjust countX and countY to be multiples of wgSizeX and wgSizeY
        std::size_t countX = ( ( imgWidth + wgSizeX - 1 ) / wgSizeX ) * wgSizeX;
        std::size_t countY = ( ( imgHeight + wgSizeY - 1 ) / wgSizeY ) * wgSizeY;
        std::size_t count = countX * countY;       // Overall number of elements
        // std::size_t size = count * sizeof ( float ); // Size of data in bytes

        cout << "Count x:" << countX << " Count Y: " << countY << endl;

        cl::size_t<3> origin;
        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;
        cl::size_t<3> region;
        region[0] = countX;
        region[1] = countY;
        region[2] = 1;

        // Allocate space for output data on the device
        cl::Image2D d_output ( context, CL_MEM_READ_WRITE,
                               cl::ImageFormat ( CL_R, CL_FLOAT ), countX, countY );

        // Allocate the host memory space for the output and initialize the device-side image with it
        Mat h_outputGpu ( countY, countX, CV_32F );

        cl::Event copy1;
        cl::Image2D image;
        image = cl::Image2D ( context, CL_MEM_READ_WRITE,
                              cl::ImageFormat ( CL_R, CL_FLOAT ), countX, countY );

        // Transfer input image data from host to device by writing it to an OpenCL Image2D object
        queue.enqueueWriteImage ( image, true, origin, region,
                                  countX * sizeof ( float ), 0, imgVector.data(), NULL,
                                  &copy1 );
        cout << "Convert Mat to image2D successfully" << endl;

        Mat h_intermediate_blur ( countY, countX, CV_32F );
        Mat h_intermediate_sbl ( countY, countX, CV_32F );
        Mat h_intermediate_nms ( countY, countX, CV_32F );
        Mat h_intermediate_strong ( countY, countX, CV_32F );
        Mat h_intermediate_weak ( countY, countX, CV_32F );

        // Create a kernel object
        string kernelblur = "GaussianBlur";
        string kernel0 = "Sobel";
        string kernel1 = "NonMaximumSuppression";
        string kernel2 = "DoubleThresholding";
        string kernel3 = "Hysteresis";

        cl::Kernel gbKernel ( program, kernelblur.c_str() );
        cl::Kernel sblKernel ( program, kernel0.c_str() );
        cl::Kernel nmsKernel ( program, kernel1.c_str() );
        cl::Kernel dtKernel ( program, kernel2.c_str() );
        cl::Kernel hKernel ( program, kernel3.c_str() );

        cl::ImageFormat format ( CL_R, CL_FLOAT );
        cl::Image2D bufferGBtoSBL ( context, CL_MEM_READ_WRITE, format, countX, countY );
        cl::Image2D bufferSBLtoNMS ( context, CL_MEM_READ_WRITE, format, countX, countY ); //Output for NMS
        cl::Image2D bufferNMStoDT ( context, CL_MEM_READ_WRITE, format, countX, countY ); // Output of the NonMaximumSuppression kernel and input to the DoubleThresholding

        cl::Image2D bufferDT_strong_toH ( context, CL_MEM_READ_WRITE, format, countX, countY ); // Output of the DoubleThresholding kernel -- strong, and input to the Hysteresis
        cl::Image2D bufferDT_weak_toH ( context, CL_MEM_READ_WRITE, format, countX, countY ); // Output of the DoubleThresholding kernel -- weak, and input to the Hysteresis

        // Launch kernel on the device
        cl::Event eventGB, eventSBL, eventNMS, eventDT, eventH;

        const float mask[] = {
            1.0f / 16, 2.0f / 16, 1.0f / 16,
            2.0f / 16, 4.0f / 16, 2.0f / 16,
            1.0f / 16, 2.0f / 16, 1.0f / 16
            };
        const int maskSize = 1; // For a 3x3 mask, maskSize would be 1.
        // Create a buffer for the mask on the OpenCL device.
        cl::Buffer maskBuffer = cl::Buffer ( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof ( mask ), const_cast<float*> ( mask ) );
        gbKernel.setArg<cl::Image2D> ( 0, image );
        gbKernel.setArg<cl::Buffer> ( 1, maskBuffer );
        gbKernel.setArg<cl::Image2D> ( 2, bufferGBtoSBL );
        gbKernel.setArg<cl_int> ( 3, maskSize );

        // set SBL Kernel arguments
        sblKernel.setArg<cl::Image2D> ( 0, bufferGBtoSBL );
        sblKernel.setArg<cl::Image2D> ( 1, bufferSBLtoNMS );

        // Set NMS Kernel arguments
        nmsKernel.setArg<cl::Image2D> ( 0, bufferSBLtoNMS );
        nmsKernel.setArg<cl::Image2D> ( 1, bufferNMStoDT ); // Output used as input for the next kernel

        // Set DT Kernel arguments
        float magMax = 0.2f;
        float magMin = 0.1f;
        dtKernel.setArg<cl::Image2D> ( 0, bufferNMStoDT ); // Output from the previous kernel
        dtKernel.setArg<cl::Image2D> ( 1, bufferDT_strong_toH ); // Output strong used as input for the next kernel
        dtKernel.setArg<cl::Image2D> ( 2, bufferDT_weak_toH ); // Output weak used as input for the next kernel
        dtKernel.setArg<cl_float> ( 3, magMax );
        dtKernel.setArg<cl_float> ( 4, magMin );

        // Set H Kernel arguments
        hKernel.setArg<cl::Image2D> ( 0, bufferDT_strong_toH );
        hKernel.setArg<cl::Image2D> ( 1, bufferDT_weak_toH );
        hKernel.setArg<cl::Image2D> ( 2, d_output ); // Output from the previous kernel

        // Gaussian Blur Kernel ------------------------------------------------
        queue.enqueueNDRangeKernel ( gbKernel, cl::NullRange,
                                     cl::NDRange ( countX, countY ),
                                     cl::NDRange ( wgSizeX, wgSizeY ), NULL, &eventGB );
        eventGB.wait();

        // store Blur output image back to the host
        queue.enqueueReadImage ( bufferGBtoSBL, true, origin, region,
                                 countX * sizeof ( float ), 0, h_intermediate_blur.data, NULL );
        cv::Mat displayBlur;
        cv::normalize ( h_intermediate_blur, displayBlur, 0, 255, cv::NORM_MINMAX );
        displayBlur.convertTo ( displayBlur, CV_8U );

        // Sobel Kernel ------------------------------------------------------------------------
        queue.enqueueNDRangeKernel ( sblKernel, cl::NullRange,
                                     cl::NDRange ( countX, countY ),
                                     cl::NDRange ( wgSizeX, wgSizeY ), NULL, &eventSBL );
        eventSBL.wait();

        // store Sobel output image back to the host
        queue.enqueueReadImage ( bufferSBLtoNMS, true, origin, region,
                                 countX * sizeof ( float ), 0, h_intermediate_sbl.data, NULL );
        double minVal, maxVal;
        cv::minMaxLoc ( h_intermediate_sbl, &minVal, &maxVal );
        cv::Mat displayImgSBL;
        cv::normalize ( h_intermediate_sbl, displayImgSBL, 0, 255, cv::NORM_MINMAX );
        displayImgSBL.convertTo ( displayImgSBL, CV_8U );

        // Non-Maximum Suppression Kernel -------------------------------------------------------
        queue.enqueueNDRangeKernel ( nmsKernel, cl::NullRange,
                                     cl::NDRange ( countX, countY ),
                                     cl::NDRange ( wgSizeX, wgSizeY ), NULL, &eventNMS );
        eventNMS.wait(); // Wait for the NMS kernel to complete

        // store Non-Maximum Suppression output image back to host
        queue.enqueueReadImage ( bufferNMStoDT, true, origin, region,
                                 countX * sizeof ( float ), 0, h_intermediate_nms.data, NULL );
        double minVal1, maxVal1;
        cv::minMaxLoc ( h_intermediate_nms, &minVal1, &maxVal1 );
        // std::cout << "NMS Min: " << minVal1 << ", Max: " << maxVal1 << std::endl;
        cv::Mat displayImgNMS;
        cv::normalize ( h_intermediate_nms, displayImgNMS, 0, 255, cv::NORM_MINMAX );
        displayImgNMS.convertTo ( displayImgNMS, CV_8U );

        // Double Thresholding Kernel -----------------------------------------------------------
        queue.enqueueNDRangeKernel ( dtKernel, cl::NullRange,
                                     cl::NDRange ( countX, countY ),
                                     cl::NDRange ( wgSizeX, wgSizeY ), NULL, &eventDT );
        eventDT.wait(); // Wait for the Double Thresholding kernel to complete

        // store Strong Image output image back to host
        queue.enqueueReadImage ( bufferDT_strong_toH, true, origin, region,
                                 countX * sizeof ( float ), 0, h_intermediate_strong.data, NULL );
        double minVal2, maxVal2;
        cv::minMaxLoc ( h_intermediate_strong, &minVal2, &maxVal2 );
        cv::Mat displayImgStrong;
        cv::normalize ( h_intermediate_strong, displayImgStrong, 0, 255, cv::NORM_MINMAX );
        displayImgStrong.convertTo ( displayImgStrong, CV_8U );

        // store Weak Image output image back to host
        queue.enqueueReadImage ( bufferDT_weak_toH, true, origin, region,
                                 countX * sizeof ( float ), 0, h_intermediate_weak.data, NULL );
        double minVal3, maxVal3;
        cv::minMaxLoc ( h_intermediate_weak, &minVal3, &maxVal3 );
        // std::cout << "weak Min: " << minVal3 << ", Max: " << maxVal3 << std::endl;
        cv::Mat displayImgWeak;
        cv::normalize ( h_intermediate_weak, displayImgWeak, 0, 255, cv::NORM_MINMAX );
        displayImgWeak.convertTo ( displayImgWeak, CV_8U );

        // Hysteresis Kernel --------------------------------------------------------------------
        queue.enqueueNDRangeKernel ( hKernel, cl::NullRange,
                                     cl::NDRange ( countX, countY ),
                                     cl::NDRange ( wgSizeX, wgSizeY ), NULL, &eventH );

        eventH.wait(); // Wait for the Hysteresis kernel to complete

        // Copy output data back to host
        cl::Event copy2;
        queue.enqueueReadImage ( d_output, true, origin, region,
                                 countX * sizeof ( float ), 0, h_outputGpu.data, NULL, &copy2 );
        double minVal4, maxVal4;
        cv::minMaxLoc ( h_outputGpu, &minVal4, &maxVal4 );
        cv::Mat FinalOutput;
        cv::normalize ( h_outputGpu, FinalOutput, 0, 255, cv::NORM_MINMAX );
        FinalOutput.convertTo ( FinalOutput, CV_8U );
        // --------------------------------------------------------

        // Print performance data
        Core::TimeSpan cpuTime = cpuEnd - cpuStart;
        Core::TimeSpan gpuTime = OpenCL::getElapsedTime ( eventNMS ) + OpenCL::getElapsedTime ( eventDT ) + OpenCL::getElapsedTime ( eventH );
        Core::TimeSpan copyTime = OpenCL::getElapsedTime ( copy1 ) + OpenCL::getElapsedTime ( copy2 );
        Core::TimeSpan overallGpuTime = gpuTime + copyTime;

        cout << "CPU Time: " << cpuTime.toString() << ", "
             << ( count / cpuTime.getSeconds() / 1e6 ) << " MPixel/s"
             << endl;
        cout << "Memory copy Time: " << copyTime.toString() << endl;
        cout << "GPU Time w/o memory copy: " << gpuTime.toString()
             << " (speedup = " << ( cpuTime.getSeconds() / gpuTime.getSeconds() )
             << ", " << ( count / gpuTime.getSeconds() / 1e6 ) << " MPixel/s)"
             << endl;
        cout << "GPU Time with memory copy: " << overallGpuTime.toString()
             << " (speedup = "
             << ( cpuTime.getSeconds() / overallGpuTime.getSeconds() ) << ", "
             << ( count / overallGpuTime.getSeconds() / 1e6 ) << " MPixel/s)"
             << endl;

        chooseStageToDisplay ( &img, &displayBlur, &displayImgSBL, &displayImgNMS, &displayImgStrong, &displayImgWeak, &FinalOutput );

        cout << "Use another image (any key) or leave (q)?" << endl;
        cout << "Your decision: ";
        cin >> decision;
        }
    while ( decision != 'q' );
    return 0;
    }


// Function to select an image
std::string selectImage() {
    char imgChoice;
    string imgName;

    do {
        cout << "-------------------------" << endl;
        cout << "Please select an image: " << endl;
        cout << "1. test1.png" << endl;
        cout << "2. lena.png" << endl;
        cout << "3. nebula.png" << endl;
        cout << "4. 4k_in.jpg (large size)" << endl;
        cout << "5. 8k_in.jpg (large size)" << endl;
        cout << "q to exit" << endl;
        cout << "-------------------------" << endl;
        cout << "Your selection: ";
        cin >> imgChoice;

        switch ( imgChoice ) {
            case '1':
                return "test1.png";
            case '2':
                return "lena.png";
            case '3':
                return "nebula.png";
            case '4':
                return "4k_in.jpg";
            case '5':
                return "8k_in.jpg";
            case 'q':
                return "q";
            default:
                cout << "Invalid choice. Please choose again." << endl;
                break; // The loop will continue due to the condition.
            }
        }
    while ( imgChoice != 'q' );

    return ""; // In case 'q' is entered.
    }

Mat NonMaximumSuppression ( const Mat& magnitude, const Mat& blurred, const Mat& angle ) {
    Mat result = magnitude.clone();
    int neighbor1X, neighbor1Y, neighbor2X, neighbor2Y;
    float gradientAngle;

    for ( int x = 0; x < blurred.rows; x++ ) {
        for ( int y = 0; y < blurred.cols; y++ ) {
            gradientAngle = angle.at<float> ( x, y );

            // Normalize angle to be in the range [0, 180)
            gradientAngle = fmodf ( fabs ( gradientAngle ), 180.0f );

            // Determine neighbors based on gradient angle
            if ( gradientAngle <= 22.5f || gradientAngle > 157.5f ) {
                neighbor1X = x - 1;
                neighbor1Y = y;
                neighbor2X = x + 1;
                neighbor2Y = y;
                }
            else if ( gradientAngle <= 67.5f ) {
                neighbor1X = x - 1;
                neighbor1Y = y - 1;
                neighbor2X = x + 1;
                neighbor2Y = y + 1;
                }
            else if ( gradientAngle <= 112.5f ) {
                neighbor1X = x;
                neighbor1Y = y - 1;
                neighbor2X = x;
                neighbor2Y = y + 1;
                }
            else {
                neighbor1X = x - 1;
                neighbor1Y = y + 1;
                neighbor2X = x + 1;
                neighbor2Y = y - 1;
                }

            // Check bounds of neighbor1
            if ( neighbor1X >= 0 && neighbor1X < blurred.rows && neighbor1Y >= 0 && neighbor1Y < blurred.cols ) {
                if ( result.at<float> ( x, y ) < result.at<float> ( neighbor1X, neighbor1Y ) ) {
                    result.at<float> ( x, y ) = 0;
                    continue;
                    }
                }

            // Check bounds of neighbor2
            if ( neighbor2X >= 0 && neighbor2X < blurred.rows && neighbor2Y >= 0 && neighbor2Y < blurred.cols ) {
                if ( result.at<float> ( x, y ) < result.at<float> ( neighbor2X, neighbor2Y ) ) {
                    result.at<float> ( x, y ) = 0;
                    continue;
                    }
                }
            }
        }
    return result;
    }

void DoubleThresholding ( const Mat& magnitude, Mat& strong, Mat& weak ) {
    // apply double thresholding
    float magMax = 0.2, magMin = 0.1;
    float gradientMagnitude;
    for ( int x = 0; x < magnitude.rows; x++ ) {
        for ( int y = 0; y < magnitude.cols; y++ ) {
            gradientMagnitude = magnitude.at<float> ( x, y );

            if ( gradientMagnitude > magMax ) {
                strong.at<float> ( x, y ) = gradientMagnitude;
                }
            else if ( gradientMagnitude <= magMax && gradientMagnitude > magMin ) {
                weak.at<float> ( x, y ) = gradientMagnitude;
                };
            }
        }
    }

Mat Hysteresis ( Mat& strong, const Mat& weak ) {
    for ( int x = 0; x < strong.rows; x++ ) {
        for ( int y = 0; y < strong.cols; y++ ) {
            if ( weak.at<float> ( x, y ) != 0 ) {
                if ( ( x + 1 < strong.rows && strong.at<float> ( x + 1, y ) != 0 ) ||
                        ( x - 1 >= 0 && strong.at<float> ( x - 1, y ) != 0 ) ||
                        ( y + 1 < strong.cols && strong.at<float> ( x, y + 1 ) ) != 0 ||
                        ( y - 1 >= 0 && strong.at<float> ( x, y - 1 ) != 0 ) ||
                        ( x - 1 >= 0 && y - 1 >= 0 && strong.at<float> ( x - 1, y - 1 ) != 0 ) ||
                        ( x + 1 < strong.rows && y - 1 >= 0 && strong.at<float> ( x + 1, y - 1 ) != 0 ) ||
                        ( x - 1 >= 0 && y + 1 < strong.cols && strong.at<float> ( x - 1, y + 1 ) != 0 ) ||
                        ( x + 1 < strong.rows && y + 1 < strong.cols && strong.at<float> ( x + 1, y + 1 ) != 0 ) ) {
                    strong.at<float> ( x, y ) = strong.at<float> ( x, y );
                    }
                }
            }
        }
    return strong;
    }

int cpuFunction ( const Mat& img ) {
    //Gaussian Blur
    Mat blurred;
    blur ( img, blurred, Size ( 3,3 ) );
    int rows = blurred.rows;
    int cols = blurred.cols;

    cout << rows << " " << cols << endl;

    // Compute image gradient
    Mat xComponent, yComponent;
    Sobel ( blurred, xComponent, CV_32F, 1, 0, 3 );
    Sobel ( blurred, yComponent, CV_32F, 0, 1, 3 );

    // Convert to polar coordinates
    Mat magnitude, angle;
    cartToPolar ( xComponent, yComponent, magnitude, angle, true );

    // Normalize values
    normalize ( magnitude, magnitude, 0, 1, NORM_MINMAX );

    // Apply non-maximum suppression
    Mat suppressed = NonMaximumSuppression ( magnitude, blurred, angle );

    // Apply double thresholding
    Mat strong = Mat::zeros ( magnitude.rows, magnitude.cols, CV_32F );
    Mat weak = Mat::zeros ( magnitude.rows, magnitude.cols, CV_32F );
    DoubleThresholding ( suppressed, strong, weak );

    // Apply hysteresis
    Mat finalEdges = Hysteresis ( strong, weak );

    return 0;
    }

void displayImage ( const string& windowName, const Mat& image ) {
    imshow ( windowName, image );
    cout << "Click any key to exit" << endl;
    waitKey ( 0 );
    destroyWindow ( windowName ); // Close the specific window
    }

void chooseStageToDisplay ( const Mat* GPtr,
                            const Mat* BlPtr,
                            const Mat* sblPtr,
                            const Mat* NMSPtr,
                            const Mat* DTSPtr,
                            const Mat* DTWPtr,
                            const Mat* FinalPtr ) {

    cout << "------------------------------------" << endl;
    cout << "* Choose the stage to display:     *" << endl;
    cout << "* 1. Gray Image                    *" << endl;
    cout << "* 2. Blurred                       *" << endl;
    cout << "* 3. Sobel                         *" << endl;
    cout << "* 4. Non-Maximum Suppression       *" << endl;
    cout << "* 5. Double Thresholding - Strong  *" << endl;
    cout << "* 6. Double Thresholding - Weak    *" << endl;
    cout << "* 7. Final Image                   *" << endl;
    cout << "* 8. Display all images            *" << endl;
    cout << "* q. Quit                          *" << endl;
    cout << "------------------------------------" << endl;
    cout << endl;

    char choice;
    do {
        cout << "Enter stage number (or 'q' to quit): ";
        cin >> choice;

        if ( !strchr ( "12345678q", choice ) ) {
            cout << "Invalid choice. Please choose again." << endl;
            }
        else {
            switch ( choice ) {
                case '1':
                    displayImage ( "Gray Image", *GPtr );
                    break;
                case '2':
                    displayImage ( "Blurred", *BlPtr );
                    break;
                case '3':
                    displayImage ( "Sobel", *sblPtr );
                    break;
                case '4':
                    displayImage ( "NonMaximumSuppression", *NMSPtr );
                    break;
                case '5':
                    displayImage ( "DoubleThresholding-Strong", *DTSPtr );
                    break;
                case '6':
                    displayImage ( "DoubleThresholding-Weak", *DTWPtr );
                    break;
                case '7':
                    displayImage ( "Final Image", *FinalPtr );
                    break;
                case '8':
                    imshow ( "Gray Image", *GPtr );
                    imshow ( "Blurred", *BlPtr );
                    imshow ( "Sobel", *sblPtr );
                    imshow ( "NonMaximumSuppression", *NMSPtr );
                    imshow ( "DoubleThresholding-Strong", *DTSPtr );
                    imshow ( "DoubleThresholding-Weak", *DTWPtr );
                    imshow ( "Final Image", *FinalPtr );
                    cout << "Click any key to exit" << endl;
                    waitKey ( 0 );
                    destroyAllWindows();
                    break;
                }
            }
        }
    while ( choice != 'q' );
    }

