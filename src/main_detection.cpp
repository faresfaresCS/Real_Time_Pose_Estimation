// C++
#include <iostream>
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
// PnP Tutorial
#include "Mesh.h"
#include "Model.h"
#include "PnPProblem.h"
#include "RobustMatcher.h"
#include "ModelRegistration.h"
#include "Utils.h"

/**  GLOBAL VARIABLES  **/

using namespace cv;
using namespace std;

/**  Functions headers  **/
void help();
void initKalmanFilter( KalmanFilter &KF, int nStates, int nMeasurements, int nInputs, double dt);
void predictKalmanFilter( KalmanFilter &KF, Mat &translation_predicted, Mat &rotation_predicted );
void updateKalmanFilter( KalmanFilter &KF, Mat &measurements,
                         Mat &translation_estimated, Mat &rotation_estimated );
void fillMeasurements( Mat &measurements,
                       const Mat &translation_measured, const Mat &rotation_measured);

/**  Main program  **/ 
int main(int argc, char *argv[])
{
    help(); 

    const String keys =
            "{help h            |      | print this message                                                 }"
            "{video v           |      | path to recorded video                                             }"
            "{model             |      | path to yml model                                                  }"
            "{mesh              |      | path to ply mesh                                                   }"
            "{keypoints k       |500    | number of keypoints to detect                                      }"
            "{ratio r           |0.7   | threshold for ratio test                                           }"
            "{iterations it     |300   | RANSAC maximum iterations count                                    }"
            "{error e           |8.0   | RANSAC reprojection error                                          }"
            "{confidence c      |0.95  | RANSAC confidence                                                  }"
            "{inliers in        |0     | minimum inliers for Kalman update                                  }"
            "{method  pnp       |0     | PnP method: (0) ITERATIVE - (1) EPNP - (2) P3P - (3) DLS - (5) AP3P}"
            "{fast f            |true  | use of robust fast match                                           }"
            "{feature           |ORB   | feature name (ORB, KAZE, AKAZE, BRISK, SIFT, SURF, BINBOOST, VGG)  }"
            "{FLANN             |false | use FLANN library for descriptors matching                         }"
            "{save              |      | path to the directory where to save the image results              }"
            "{displayFiltered   |false | display filtered pose (from Kalman filter)                         }"
            ;
    CommandLineParser parser(argc, argv, keys);
    string yml_read_path[2] = {"",""};
    string ply_read_path[2] = {"",""};
    string video_read_path = samples::findFile("samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/Data/test.mp4");   // recorded video
    yml_read_path[0] = samples::findFile("samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/Data/box1.yml"); // 3dpts + descriptors
    yml_read_path[1] = samples::findFile("samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/Data/box2.yml"); // 3dpts + descriptors
    ply_read_path[0] = samples::findFile("samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/Data/box1.ply");         // mesh
    ply_read_path[1] = samples::findFile("samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/Data/box2.ply");         // mesh

    // double params_WEBCAM[] = { width*f/sx,   // fx
    //                            height*f/sy,  // fy
    //                            width/2,      // cx
    //                            height/2};    // cy

	const double params_WEBCAM[] = {1.3923167100834285e+03,
                              		1.3967840677779698e+03,
                          			960,
                              		540};
    
    // Some basic colors
    Scalar red(0, 0, 255);
    Scalar green(0,255,0);
    Scalar blue(255,0,0);
    Scalar yellow(0,255,255);

    // Robust Matcher parameters
    int numKeyPoints = 500;      // number of detected keypoints
    float ratioTest = 0.7f;      // ratio test
    bool fast_match = true;       // fastRobustMatch() or robustMatch()

    // RANSAC parameters
    int iterationsCount = 300;      // number of Ransac iterations.
    float reprojectionError = 8.0;  // maximum allowed distance to consider it an inlier.
    double confidence = 0.95;       // ransac successful confidence.

    // Kalman Filter parameters
    int minInliersKalman = 30;    // Kalman threshold updating

    // PnP parameters
    int pnpMethod = SOLVEPNP_ITERATIVE;
    string featureName = "ORB";
    bool useFLANN = false;

    // Save results
    string saveDirectory = "";
    Mat frameSave;
    int frameCount = 0;

    bool displayFilteredPose = false;

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    else
    {
        video_read_path = parser.get<string>("video").size() > 0 ? parser.get<string>("video") : video_read_path;
        yml_read_path[0] = parser.get<string>("model").size() > 0 ? parser.get<string>("model") : yml_read_path[0];
        ply_read_path[0] = parser.get<string>("mesh").size() > 0 ? parser.get<string>("mesh") : ply_read_path[0];
        yml_read_path[1] = parser.get<string>("model").size() > 0 ? parser.get<string>("model") : yml_read_path[1];
        ply_read_path[1] = parser.get<string>("mesh").size() > 0 ? parser.get<string>("mesh") : ply_read_path[1];
        numKeyPoints = parser.has("keypoints") ? parser.get<int>("keypoints") : numKeyPoints;
        ratioTest = parser.has("ratio") ? parser.get<float>("ratio") : ratioTest;
        fast_match = parser.has("fast") ? parser.get<bool>("fast") : fast_match;
        iterationsCount = parser.has("iterations") ? parser.get<int>("iterations") : iterationsCount;
        reprojectionError = parser.has("error") ? parser.get<float>("error") : reprojectionError;
        confidence = parser.has("confidence") ? parser.get<float>("confidence") : confidence;
        minInliersKalman = parser.has("inliers") ? parser.get<int>("inliers") : minInliersKalman;
        pnpMethod = parser.has("method") ? parser.get<int>("method") : pnpMethod;
        featureName = parser.has("feature") ? parser.get<string>("feature") : featureName;
        useFLANN = parser.has("FLANN") ? parser.get<bool>("FLANN") : useFLANN;
        saveDirectory = parser.has("save") ? parser.get<string>("save") : saveDirectory;
        displayFilteredPose = parser.has("displayFiltered") ? parser.get<bool>("displayFiltered") : displayFilteredPose;
    }

    std::cout << "Video: " << video_read_path << std::endl;
    std::cout << "Training data1: " << yml_read_path[0] << std::endl;
    std::cout << "CAD model1: " << ply_read_path[0] << std::endl;
    std::cout << "Training data2: " << yml_read_path[1] << std::endl;
    std::cout << "CAD model2: " << ply_read_path[1] << std::endl;
    std::cout << "Ratio test threshold: " << ratioTest << std::endl;
    std::cout << "Fast match(no symmetry test)?: " << fast_match << std::endl;
    std::cout << "RANSAC number of iterations: " << iterationsCount << std::endl;
    std::cout << "RANSAC reprojection error: " << reprojectionError << std::endl;
    std::cout << "RANSAC confidence threshold: " << confidence << std::endl;
    std::cout << "Kalman number of inliers: " << minInliersKalman << std::endl;
    std::cout << "PnP method: " << pnpMethod << std::endl;
    std::cout << "Feature: " << featureName << std::endl;
    std::cout << "Number of keypoints for ORB: " << numKeyPoints << std::endl;
    std::cout << "Use FLANN-based matching? " << useFLANN << std::endl;
    std::cout << "Save directory: " << saveDirectory << std::endl;
    std::cout << "Display filtered pose from Kalman filter? " << displayFilteredPose << std::endl;

    PnPProblem pnp_detection(params_WEBCAM);
    PnPProblem pnp_detection_est(params_WEBCAM);

    Model model1;               // instantiate Model object
    model1.load(yml_read_path[0]); // load a 3D textured object model

    Mesh mesh1;                 // instantiate Mesh object
    mesh1.load(ply_read_path[0]);  // load an object mesh

    Model model2;               // instantiate Model object
    model2.load(yml_read_path[1]); // load a 3D textured object model

    Mesh mesh2;                 // instantiate Mesh object
    mesh2.load(ply_read_path[1]);  // load an object mesh

    RobustMatcher rmatcher1;
    RobustMatcher rmatcher2;                                                     // instantiate RobustMatcher

    Ptr<FeatureDetector> detector1, descriptor1;
    createFeatures(featureName, numKeyPoints, detector1, descriptor1);
    rmatcher1.setFeatureDetector(detector1);                                      // set feature detector
    rmatcher1.setDescriptorExtractor(descriptor1);                                // set descriptor extractor
    rmatcher1.setDescriptorMatcher(createMatcher(featureName, useFLANN));        // set matcher
    rmatcher1.setRatio(ratioTest); // set ratio test parameter
    if (!model1.get_trainingImagePath().empty())
    {
        Mat trainingImg = imread(model1.get_trainingImagePath());
        rmatcher1.setTrainingImage(trainingImg);
    }
    
    Ptr<FeatureDetector> detector2, descriptor2;
    createFeatures(featureName, numKeyPoints, detector2, descriptor2);
    rmatcher2.setFeatureDetector(detector2);                                      // set feature detector
    rmatcher2.setDescriptorExtractor(descriptor2);                                // set descriptor extractor
    rmatcher2.setDescriptorMatcher(createMatcher(featureName, useFLANN));        // set matcher
    rmatcher2.setRatio(ratioTest); // set ratio test parameter
    if (!model2.get_trainingImagePath().empty())
    {
        Mat trainingImg = imread(model2.get_trainingImagePath());
        rmatcher2.setTrainingImage(trainingImg);
    }
    

    KalmanFilter KF;             // instantiate Kalman Filter
    int nStates = 18;            // the number of states
    int nMeasurements = 6;       // the number of measured states
    int nInputs = 0;             // the number of control actions
    double dt = 0.125;           // time between measurements (1/FPS)

    initKalmanFilter(KF, nStates, nMeasurements, nInputs, dt);    // init function
    Mat measurements(nMeasurements, 1, CV_64FC1); measurements.setTo(Scalar(0));
    bool good_measurement = false;

    // Get the MODEL 1 INFO
    vector<Point3f> list_points3d_model1 = model1.get_points3d();  // list with model 3D coordinates
    Mat descriptors_model1 = model1.get_descriptors();             // list with descriptors of each 3D coordinate
    vector<KeyPoint> keypoints_model1 = model1.get_keypoints();
    
    // Get the MODEL 2 INFO
    vector<Point3f> list_points3d_model2 = model2.get_points3d();  // list with model 3D coordinates
    Mat descriptors_model2 = model2.get_descriptors();             // list with descriptors of each 3D coordinate
    vector<KeyPoint> keypoints_model2 = model2.get_keypoints();

    // Create & Open Window
    namedWindow("REAL TIME DEMO", WINDOW_KEEPRATIO);

    VideoCapture cap;                           // instantiate VideoCapture
    cap.open(video_read_path);                  // open a recorded video
    //cap.open(0); 
    if(!cap.isOpened())   // check if we succeeded
    {
        cout << "Could not open the camera device" << endl;
        return -1;
    }

    if (!saveDirectory.empty())
    {
        if (!cv::utils::fs::exists(saveDirectory))
        {
            std::cout << "Create directory: " << saveDirectory << std::endl;
            cv::utils::fs::createDirectories(saveDirectory);
        }
    }

    // Measure elapsed time
    TickMeter tm;

    Mat frame, frame_vis, frame_matching1, frame_matching2;
    while(cap.read(frame) && (char)waitKey(30) != 27) // capture frame until ESC is pressed
    {
        tm.reset();
        tm.start();
        frame_vis = frame.clone();    // refresh visualisation frame

        // -- Step 1: Robust matching between model descriptors and scene descriptors
        vector<DMatch> good_matches1;       // to obtain the 3D points of the model
        vector<KeyPoint> keypoints_scene1;  // to obtain the 2D points of the scene
        
        // -- Step 1: Robust matching between model descriptors and scene descriptors
        vector<DMatch> good_matches2;       // to obtain the 3D points of the model
        vector<KeyPoint> keypoints_scene2;  // to obtain the 2D points of the scene

        if(fast_match)
        {
            rmatcher1.fastRobustMatch(frame, good_matches1, keypoints_scene1, descriptors_model1, keypoints_model1);
            rmatcher2.fastRobustMatch(frame, good_matches2, keypoints_scene2, descriptors_model2, keypoints_model2);
        }
        else
        {
            rmatcher1.robustMatch(frame, good_matches1, keypoints_scene1, descriptors_model1, keypoints_model1);
            rmatcher2.robustMatch(frame, good_matches2, keypoints_scene2, descriptors_model2, keypoints_model2);
        }

        frame_matching1 = rmatcher1.getImageMatching();
        frame_matching2 = rmatcher2.getImageMatching();
        if (!frame_matching1.empty() || !frame_matching2.empty())
        {
            //imshow("Keypoints matching object 1", frame_matching1);
            //imshow("Keypoints matching object 2", frame_matching2);
        }

        // -- Step 2: Find out the 2D/3D correspondences
        vector<Point3f> list_points3d_model_match1; // container for the model 3D coordinates found in the scene
        vector<Point2f> list_points2d_scene_match1; // container for the model 2D coordinates found in the scene
        
        // -- Step 2: Find out the 2D/3D correspondences
        vector<Point3f> list_points3d_model_match2; // container for the model 3D coordinates found in the scene
        vector<Point2f> list_points2d_scene_match2; // container for the model 2D coordinates found in the scene

        for(unsigned int match_index = 0; match_index < good_matches1.size(); ++match_index)
        {
            Point3f point3d_model1 = list_points3d_model1[ good_matches1[match_index].trainIdx ];  // 3D point from model
            Point2f point2d_scene1 = keypoints_scene1[ good_matches1[match_index].queryIdx ].pt; // 2D point from the scene
            list_points3d_model_match1.push_back(point3d_model1);         // add 3D point
            list_points2d_scene_match1.push_back(point2d_scene1);         // add 2D point
         }
         
         for(unsigned int match_index = 0; match_index < good_matches2.size(); ++match_index)
         {   
            Point3f point3d_model2 = list_points3d_model2[ good_matches2[match_index].trainIdx ];  // 3D point from model
            Point2f point2d_scene2 = keypoints_scene2[ good_matches2[match_index].queryIdx ].pt; // 2D point from the scene
            list_points3d_model_match2.push_back(point3d_model2);         // add 3D point
            list_points2d_scene_match2.push_back(point2d_scene2);         // add 2D point
         }

        // Draw outliers
        //draw2DPoints(frame_vis, list_points2d_scene_match1, red);
        
        // Draw outliers
        //draw2DPoints(frame_vis, list_points2d_scene_match2, yellow);

        Mat inliers_idx1;
        vector<Point2f> list_points2d_inliers1;
        
        Mat inliers_idx2;
        vector<Point2f> list_points2d_inliers2;

        // Instantiate estimated translation and rotation
        good_measurement = false;

        if(good_matches1.size() >= 4 || good_matches2.size() >= 4) // OpenCV requires solvePnPRANSAC to minimally have 4 set of points
        {
            // -- Step 3: Estimate the pose using RANSAC approach
            if(good_matches1.size() >= 4){
            pnp_detection.estimatePoseRANSAC( list_points3d_model_match1, list_points2d_scene_match1,
                                              pnpMethod, inliers_idx1,
                                              iterationsCount, reprojectionError, confidence );
            }
            if(good_matches2.size() >= 4){    
            pnp_detection.estimatePoseRANSAC( list_points3d_model_match2, list_points2d_scene_match2,
                                              pnpMethod, inliers_idx2,
                                              iterationsCount, reprojectionError, confidence );
        	}
            // -- Step 4: Catch the inliers keypoints to draw
            for(int inliers_index = 0; inliers_index < inliers_idx1.rows; ++inliers_index)
            {
                int n = inliers_idx1.at<int>(inliers_index);         // i-inlier
                Point2f point2d = list_points2d_scene_match1[n];     // i-inlier point 2D
                list_points2d_inliers1.push_back(point2d);           // add i-inlier to list
            }
            
            // -- Step 4: Catch the inliers keypoints to draw
            for(int inliers_index = 0; inliers_index < inliers_idx2.rows; ++inliers_index)
            {
                int n = inliers_idx2.at<int>(inliers_index);         // i-inlier
                Point2f point2d = list_points2d_scene_match2[n];     // i-inlier point 2D
                list_points2d_inliers2.push_back(point2d);           // add i-inlier to list
            }

            // Draw inliers points 2D
            draw2DPoints(frame_vis, list_points2d_inliers1, blue);
            draw2DPoints(frame_vis, list_points2d_inliers2, blue);

            // -- Step 5: Kalman Filter

            // GOOD MEASUREMENT
            if( inliers_idx1.rows >= minInliersKalman )
            {
                // Get the measured translation
                Mat translation_measured = pnp_detection.get_t_matrix();

                // Get the measured rotation
                Mat rotation_measured = pnp_detection.get_R_matrix();

                // fill the measurements vector
                fillMeasurements(measurements, translation_measured, rotation_measured);
                good_measurement = true;
            }
            
            if( inliers_idx2.rows >= minInliersKalman )
            {
                // Get the measured translation
                Mat translation_measured = pnp_detection.get_t_matrix();

                // Get the measured rotation
                Mat rotation_measured = pnp_detection.get_R_matrix();

                // fill the measurements vector
                fillMeasurements(measurements, translation_measured, rotation_measured);
                good_measurement = true;
            }

            // update the Kalman filter with good measurements, otherwise with previous valid measurements
            Mat translation_estimated(3, 1, CV_64FC1);
            Mat rotation_estimated(3, 3, CV_64FC1);
            updateKalmanFilter( KF, measurements,
                                translation_estimated, rotation_estimated);

            // -- Step 6: Set estimated projection matrix
            pnp_detection_est.set_P_matrix(rotation_estimated, translation_estimated);
        }

        // -- Step X: Draw pose and coordinate frame
        float l = 5;
        vector<Point2f> pose_points2d;
        if (!good_measurement || displayFilteredPose)
        {
            // drawObjectMesh(frame_vis, &mesh1, &pnp_detection_est, yellow); // draw estimated pose
            // drawObjectMesh(frame_vis, &mesh2, &pnp_detection_est, yellow); // draw estimated pose

            pose_points2d.push_back(pnp_detection_est.backproject3DPoint(Point3f(0,0,0)));  // axis center
            pose_points2d.push_back(pnp_detection_est.backproject3DPoint(Point3f(l,0,0)));  // axis x
            pose_points2d.push_back(pnp_detection_est.backproject3DPoint(Point3f(0,l,0)));  // axis y
            pose_points2d.push_back(pnp_detection_est.backproject3DPoint(Point3f(0,0,l)));  // axis z
            // draw3DCoordinateAxes(frame_vis, pose_points2d);           // draw axes
        }
        else
        {
            // drawObjectMesh(frame_vis, &mesh1, &pnp_detection, green);  // draw current pose
            // drawObjectMesh(frame_vis, &mesh2, &pnp_detection, green);  // draw current pose

            pose_points2d.push_back(pnp_detection.backproject3DPoint(Point3f(0,0,0)));  // axis center
            pose_points2d.push_back(pnp_detection.backproject3DPoint(Point3f(l,0,0)));  // axis x
            pose_points2d.push_back(pnp_detection.backproject3DPoint(Point3f(0,l,0)));  // axis y
            pose_points2d.push_back(pnp_detection.backproject3DPoint(Point3f(0,0,l)));  // axis z
            // draw3DCoordinateAxes(frame_vis, pose_points2d);           // draw axes
        }

        // FRAME RATE
        // see how much time has elapsed
        tm.stop();

        // calculate current FPS
        // double fps = 1.0 / tm.getTimeSec();

        // drawFPS(frame_vis, fps, yellow); // frame ratio
        double detection_ratio1 = ((double)inliers_idx1.rows/(double)good_matches1.size())*100;
        
        double detection_ratio2 = ((double)inliers_idx2.rows/(double)good_matches2.size())*100;

        // -- Step X: Draw some debugging text
        // Draw some debug text
        int inliers_int1 = inliers_idx1.rows;
        int outliers_int1 = (int)good_matches1.size() - inliers_int1;
        string inliers_str1 = IntToString(inliers_int1);
        string outliers_str1 = IntToString(outliers_int1);
        string n1 = IntToString((int)good_matches1.size());
        string text = "***Item 1 Found***";
        string text5 = "Confidence:";
        string text2 = "Inliers: " + inliers_str1 + " - Outliers: " + outliers_str1;

        
        
        int inliers_int2 = inliers_idx2.rows;
        int outliers_int2 = (int)good_matches2.size() - inliers_int2;
        string inliers_str2 = IntToString(inliers_int2);
        string outliers_str2 = IntToString(outliers_int2);
        string n2 = IntToString((int)good_matches2.size());
        string text3 = "***Item 2 Found***";
        string text6 = "Confidence:";
        string text4 = "Inliers: " + inliers_str2 + " - Outliers: " + outliers_str2;

  //   	if( (detection_ratio1 + detection_ratio2) > 150){
  //   		drawText(frame_vis, text, green);
  //       	drawConfidence1(frame_vis, detection_ratio1, yellow);
  //       	drawText3(frame_vis, text3, green);
  //           drawConfidence2(frame_vis, detection_ratio2, yellow);
		// }
    	if(detection_ratio1 != 100 && detection_ratio2 != 100){
			if(detection_ratio1 > detection_ratio2){
	        	drawText(frame_vis, text, blue);
	        	drawText5(frame_vis, text5, blue);
	        	drawConfidence1(frame_vis, detection_ratio1, yellow);
	        // drawText2(frame_vis, text2, red);
	        }

	        else if(detection_ratio2 > 0){
	        	drawText3(frame_vis, text3, red);
	        	drawText6(frame_vis, text6, red);
	            drawConfidence2(frame_vis, detection_ratio2, yellow);
	        // drawText4(frame_vis, text4, red);
			}
		}
		//cv2.WINDOW_NORMAL makes the output window resizealbe
    	// namedWindow('Resized Window', WINDOW_NORMAL);

    	// //resize the window according to the screen resolution
    	// resizeWindow('Resized Window', '500', '500');

    	// imshow("Resized Window", frame_vis);
        imshow("REAL TIME DEMO", frame_vis);

        if (!saveDirectory.empty())
        {
            const int widthSave = !frame_matching1.empty() ? frame_matching1.cols : frame_vis.cols;
            const int heightSave = !frame_matching1.empty() ? frame_matching1.rows + frame_vis.rows : frame_vis.rows;
            frameSave = Mat::zeros(heightSave, widthSave, CV_8UC3);
            if (!frame_matching1.empty())
            {
                int startX = (int)((widthSave - frame_vis.cols) / 2.0);
                Mat roi = frameSave(Rect(startX, 0, frame_vis.cols, frame_vis.rows));
                frame_vis.copyTo(roi);

                roi = frameSave(Rect(0, frame_vis.rows, frame_matching1.cols, frame_matching1.rows));
                frame_matching1.copyTo(roi);
            }
            else
            {
                frame_vis.copyTo(frameSave);
            }

            string saveFilename = format(string(saveDirectory + "/image_%04d.png").c_str(), frameCount);
            imwrite(saveFilename, frameSave);
            frameCount++;
        }
    }

    // Close and Destroy Window
    destroyWindow("REAL TIME DEMO");

    cout << "GOODBYE ..." << endl;
}

/**********************************************************************************************************/
void help()
{
    cout
            << "--------------------------------------------------------------------------"   << endl
            << "This program shows how to detect an object given its 3D textured model. You can choose to "
            << "use a recorded video or the webcam."                                          << endl
            << "Usage:"                                                                       << endl
            << "./cpp-tutorial-pnp_detection -help"                                           << endl
            << "Keys:"                                                                        << endl
            << "'esc' - to quit."                                                             << endl
            << "--------------------------------------------------------------------------"   << endl
            << endl;
}

/**********************************************************************************************************/
void initKalmanFilter(KalmanFilter &KF, int nStates, int nMeasurements, int nInputs, double dt)
{
    KF.init(nStates, nMeasurements, nInputs, CV_64F);                 // init Kalman Filter

    setIdentity(KF.processNoiseCov, Scalar::all(1e-5));       // set process noise
    setIdentity(KF.measurementNoiseCov, Scalar::all(1e-2));   // set measurement noise
    setIdentity(KF.errorCovPost, Scalar::all(1));             // error covariance

    /** DYNAMIC MODEL **/

    //  [1 0 0 dt  0  0 dt2   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 1 0  0 dt  0   0 dt2   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 1  0  0 dt   0   0 dt2 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  1  0  0  dt   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  1  0   0  dt   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  1   0   0  dt 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   1   0   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   1   0 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   0   1 0 0 0  0  0  0   0   0   0]
    //  [0 0 0  0  0  0   0   0   0 1 0 0 dt  0  0 dt2   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 1 0  0 dt  0   0 dt2   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 1  0  0 dt   0   0 dt2]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  1  0  0  dt   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  1  0   0  dt   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  1   0   0  dt]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   1   0   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   1   0]
    //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   0   1]

    // position
    KF.transitionMatrix.at<double>(0,3) = dt;
    KF.transitionMatrix.at<double>(1,4) = dt;
    KF.transitionMatrix.at<double>(2,5) = dt;
    KF.transitionMatrix.at<double>(3,6) = dt;
    KF.transitionMatrix.at<double>(4,7) = dt;
    KF.transitionMatrix.at<double>(5,8) = dt;
    KF.transitionMatrix.at<double>(0,6) = 0.5*pow(dt,2);
    KF.transitionMatrix.at<double>(1,7) = 0.5*pow(dt,2);
    KF.transitionMatrix.at<double>(2,8) = 0.5*pow(dt,2);

    // orientation
    KF.transitionMatrix.at<double>(9,12) = dt;
    KF.transitionMatrix.at<double>(10,13) = dt;
    KF.transitionMatrix.at<double>(11,14) = dt;
    KF.transitionMatrix.at<double>(12,15) = dt;
    KF.transitionMatrix.at<double>(13,16) = dt;
    KF.transitionMatrix.at<double>(14,17) = dt;
    KF.transitionMatrix.at<double>(9,15) = 0.5*pow(dt,2);
    KF.transitionMatrix.at<double>(10,16) = 0.5*pow(dt,2);
    KF.transitionMatrix.at<double>(11,17) = 0.5*pow(dt,2);


    /** MEASUREMENT MODEL **/

    //  [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
    //  [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]

    KF.measurementMatrix.at<double>(0,0) = 1;  // x
    KF.measurementMatrix.at<double>(1,1) = 1;  // y
    KF.measurementMatrix.at<double>(2,2) = 1;  // z
    KF.measurementMatrix.at<double>(3,9) = 1;  // roll
    KF.measurementMatrix.at<double>(4,10) = 1; // pitch
    KF.measurementMatrix.at<double>(5,11) = 1; // yaw
}

/**********************************************************************************************************/
void updateKalmanFilter( KalmanFilter &KF, Mat &measurement,
                         Mat &translation_estimated, Mat &rotation_estimated )
{
    // First predict, to update the internal statePre variable
    Mat prediction = KF.predict();

    // The "correct" phase that is going to use the predicted value and our measurement
    Mat estimated = KF.correct(measurement);

    // Estimated translation
    translation_estimated.at<double>(0) = estimated.at<double>(0);
    translation_estimated.at<double>(1) = estimated.at<double>(1);
    translation_estimated.at<double>(2) = estimated.at<double>(2);

    // Estimated euler angles
    Mat eulers_estimated(3, 1, CV_64F);
    eulers_estimated.at<double>(0) = estimated.at<double>(9);
    eulers_estimated.at<double>(1) = estimated.at<double>(10);
    eulers_estimated.at<double>(2) = estimated.at<double>(11);

    // Convert estimated quaternion to rotation matrix
    rotation_estimated = euler2rot(eulers_estimated);
}

/**********************************************************************************************************/
void fillMeasurements( Mat &measurements,
                       const Mat &translation_measured, const Mat &rotation_measured)
{
    // Convert rotation matrix to euler angles
    Mat measured_eulers(3, 1, CV_64F);
    measured_eulers = rot2euler(rotation_measured);

    // Set measurement to predict
    measurements.at<double>(0) = translation_measured.at<double>(0); // x
    measurements.at<double>(1) = translation_measured.at<double>(1); // y
    measurements.at<double>(2) = translation_measured.at<double>(2); // z
    measurements.at<double>(3) = measured_eulers.at<double>(0);      // roll
    measurements.at<double>(4) = measured_eulers.at<double>(1);      // pitch
    measurements.at<double>(5) = measured_eulers.at<double>(2);      // yaw
}
