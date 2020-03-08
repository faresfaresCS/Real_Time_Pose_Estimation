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
int NUM_OF_BOXES = 2;

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
            "{keypoints k       |2000  | number of keypoints to detect                                      }"
            "{ratio r           |0.9   | threshold for ratio test                                           }"
            "{iterations it     |300   | RANSAC maximum iterations count                                    }"
            "{error e           |8.0   | RANSAC reprojection error                                          }"
            "{confidence c      |0.99  | RANSAC confidence                                                  }"
            "{inliers in        |0     | minimum inliers for Kalman update                                  }"
            "{method  pnp       |1     | PnP method: (0) ITERATIVE - (1) EPNP - (2) P3P - (3) DLS - (5) AP3P}"
            "{fast f            |true  | use of robust fast match                                           }"
            "{feature           |ORB   | feature name (ORB, KAZE, AKAZE, BRISK, SIFT, SURF, BINBOOST, VGG)  }"
            "{FLANN             |false | use FLANN library for descriptors matching                         }"
            "{save              |      | path to the directory where to save the image results              }"
            "{displayFiltered   |false | display filtered pose (from Kalman filter)                         }"
            ;
    CommandLineParser parser(argc, argv, keys);
    string yml_read_path[NUM_OF_BOXES] = {"",""};
    string ply_read_path[NUM_OF_BOXES] = {"",""};
    string video_read_path = samples::findFile("samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/Data/2.mkv");   // recorded video
    
    for(int i=0; i< NUM_OF_BOXES; i++){
        yml_read_path[i] = samples::findFile("samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/Data/box" + IntToString(i) + ".yml"); // 3dpts + descriptors
        ply_read_path[i] = samples::findFile("samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/Data/box"+ IntToString(i) + ".ply");  // mesh
    }
    
    // LOGITECH2
	// const double params_WEBCAM[] = {1.3923167100834285e+03,
 	//                              	1.3967840677779698e+03,
 	//                          		960,
 	// 									540};            
    // 
 	// LOGITECH1
    // const double params_WEBCAM[] = { 1.28397223e+03,
    //                             1.29943966e+03,
    //                             9.66018414e+02,
    //                             4.72880443e+02};

    const double params_WEBCAM[] = {1.34309842e+03,
    								1.34579072e+03,
    								8.01983476e+02,
    								5.84303298e+02};

    // Some basic colors
    Scalar red(0, 0, 255);
    Scalar green(0,255,0);
    Scalar blue(255,0,0);
    Scalar yellow(0,255,255);

    // Robust Matcher parameters
    int numKeyPoints = 2000;      // number of detected keypoints
    float ratioTest = 0.9f;      // ratio test
    bool fast_match = true;      // fastRobustMatch() or robustMatch()

    // RANSAC parameters
    int iterationsCount = 300;      // number of Ransac iterations.
    float reprojectionError = 12.0;  // maximum allowed distance to consider it an inlier.
    double confidence = 0.99;       // ransac successful confidence.

    // Kalman Filter parameters
    int minInliersKalman = 30;    // Kalman threshold updating

    // PnP parameters
    int pnpMethod = SOLVEPNP_EPNP;
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
        for(int i=0; i< NUM_OF_BOXES; i++){
            video_read_path = parser.get<string>("video").size() > 0 ? parser.get<string>("video") : video_read_path;
            yml_read_path[i] = parser.get<string>("model").size() > 0 ? parser.get<string>("model") : yml_read_path[i];
            ply_read_path[i] = parser.get<string>("mesh").size() > 0 ? parser.get<string>("mesh") : ply_read_path[i];
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
    }


    PnPProblem pnp_detection(params_WEBCAM);
    PnPProblem pnp_detection_est(params_WEBCAM);

    Model models[NUM_OF_BOXES]; // instantiate Model object
    Mesh meshes [NUM_OF_BOXES]; // instantiate Mesh object
    RobustMatcher rmatcher[NUM_OF_BOXES];
    
    for(int i=0; i< NUM_OF_BOXES; i++){
        models[i].load(yml_read_path[i]);  // load a 3D textured object model
        meshes[i].load(ply_read_path[i]);  // load an object mesh
    }

    Ptr<FeatureDetector> detector[NUM_OF_BOXES], descriptor[NUM_OF_BOXES];
    for(int i=0; i< NUM_OF_BOXES; i++){
        createFeatures(featureName, numKeyPoints, detector[i], descriptor[i]);
        rmatcher[i].setFeatureDetector(detector[i]);                                // set feature detector
        rmatcher[i].setDescriptorExtractor(descriptor[i]);                          // set descriptor extractor
        rmatcher[i].setDescriptorMatcher(createMatcher(featureName, useFLANN));     // set matcher
        rmatcher[i].setRatio(ratioTest);                                            // set ratio test parameter
        if (!models[i].get_trainingImagePath().empty())
        {
            Mat trainingImg = imread(models[i].get_trainingImagePath());
            rmatcher[i].setTrainingImage(trainingImg);
        }
    }    


    KalmanFilter KF;             // instantiate Kalman Filter
    int nStates = 18;            // the number of states
    int nMeasurements = 6;       // the number of measured states
    int nInputs = 0;             // the number of control actions
    double dt = 0.125;           // time between measurements (1/FPS)

    initKalmanFilter(KF, nStates, nMeasurements, nInputs, dt);    // init function
    Mat measurements(nMeasurements, 1, CV_64FC1); measurements.setTo(Scalar(0));
    bool good_measurement = false;

	vector<Point3f> list_points3d_model[NUM_OF_BOXES];
	Mat descriptors_model[NUM_OF_BOXES];
	vector<KeyPoint> keypoints_model[NUM_OF_BOXES];

    // Get the MODEL INFO
    for(int i=0; i< NUM_OF_BOXES; i++){
        list_points3d_model[i] = models[i].get_points3d();  // list with model 3D coordinates
        descriptors_model[i] = models[i].get_descriptors();             // list with descriptors of each 3D coordinate
        keypoints_model[i] = models[i].get_keypoints();
    }

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

    Mat frame, frame_vis, frame_matching[NUM_OF_BOXES];
    while(cap.read(frame) && (char)waitKey(30) != 27) // capture frame until ESC is pressed
    {
        tm.reset();
        tm.start();
        frame_vis = frame.clone();    // refresh visualisation frame

        // -- Step 1: Robust matching between model descriptors and scene descriptors
        vector<DMatch> good_matches[NUM_OF_BOXES];       // to obtain the 3D points of the model
        vector<KeyPoint> keypoints_scene[NUM_OF_BOXES];  // to obtain the 2D points of the scene

        for(int i=0;i< NUM_OF_BOXES; i++){
            if(fast_match)
            {
                rmatcher[i].fastRobustMatch(frame, good_matches[i], keypoints_scene[i], descriptors_model[i], keypoints_model[i]);
            }
            else
            {
                rmatcher[i].robustMatch(frame, good_matches[i], keypoints_scene[i], descriptors_model[i], keypoints_model[i]);
            }

            frame_matching[i] = rmatcher[i].getImageMatching();

            if (!frame_matching[i].empty())
        		{
            		//imshow("Product " + IntToString(i), frame_matching[i]);
         		}
        }

        // -- Step 2: Find out the 2D/3D correspondences
        vector<Point3f> list_points3d_model_match[NUM_OF_BOXES]; // container for the model 3D coordinates found in the scene
        vector<Point2f> list_points2d_scene_match[NUM_OF_BOXES]; // container for the model 2D coordinates found in the scene
        Point3f point3d_model[NUM_OF_BOXES];
        Point2f point2d_scene[NUM_OF_BOXES];

        for(int i=0;i< NUM_OF_BOXES; i++){
            for(unsigned int match_index = 0; match_index < good_matches[i].size(); ++match_index)
            {
                point3d_model[i] = list_points3d_model[i][ good_matches[i][match_index].trainIdx ];  // 3D point from model
                point2d_scene[i] = keypoints_scene[i][ good_matches[i][match_index].queryIdx ].pt; // 2D point from the scene
                list_points3d_model_match[i].push_back(point3d_model[i]);         // add 3D point
                list_points2d_scene_match[i].push_back(point2d_scene[i]);         // add 2D point
            }
         

	        // Draw outliers
	        draw2DPoints(frame_vis, list_points2d_scene_match[i], red);
	        
	        // Draw outliers
	        draw2DPoints(frame_vis, list_points2d_scene_match[i], yellow);
        }

        Mat inliers_idx[NUM_OF_BOXES];
        vector<Point2f> list_points2d_inliers[NUM_OF_BOXES];
        
        // Instantiate estimated translation and rotation
        good_measurement = false;

        double detection_ratio[NUM_OF_BOXES];
        int inliers_int[NUM_OF_BOXES];
        int outliers_int[NUM_OF_BOXES];
        string inliers_str[NUM_OF_BOXES];
        string outliers_str[NUM_OF_BOXES];
        //string n[NUM_OF_BOXES];
        
        for(int i=0; i<NUM_OF_BOXES; i++)
        {
            // -- Step 3: Estimate the pose using RANSAC approach
            if(good_matches[i].size() >= 4){
            pnp_detection.estimatePoseRANSAC( list_points3d_model_match[i], list_points2d_scene_match[i],
                                              pnpMethod, inliers_idx[i],
                                              iterationsCount, reprojectionError, confidence );
            }
            
            // -- Step 4: Catch the inliers keypoints to draw
            for(int inliers_index = 0; inliers_index < inliers_idx[i].rows; ++inliers_index)
            {
                int n = inliers_idx[i].at<int>(inliers_index);         // i-inlier
                Point2f point2d = list_points2d_scene_match[i][n];     // i-inlier point 2D
                list_points2d_inliers[i].push_back(point2d);           // add i-inlier to list
            }
            
            // Draw inliers points 2D
            draw2DPoints(frame_vis, list_points2d_inliers[i], blue);

            // -- Step 5: Kalman Filter

            // GOOD MEASUREMENT
            if( inliers_idx[i].rows >= minInliersKalman )
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
	        detection_ratio[i] = ((double)inliers_idx[i].rows/(double)good_matches[i].size())*100;
	        
	        // -- Step X: Draw some debugging text
	        // Draw some debug text
	        inliers_int[i] = inliers_idx[i].rows;
	        outliers_int[i] = (int)good_matches[i].size() - inliers_int[i];
	        inliers_str[i] = IntToString(inliers_int[i]);
	        outliers_str[i] = IntToString(outliers_int[i]);
	        //n[i] = IntToString((int)good_matches[i].size());
	    }
	    
	    int max_detection_ratio = 0;
	    int sum_inliers_num = 0;
	    int sum_outliers_num = 0;

	    for(int i = 0; i < NUM_OF_BOXES; i++){
	    	sum_inliers_num += inliers_int[i];
	    	sum_outliers_num += outliers_int[i];
	    	if(detection_ratio[i] == 100)
	    		detection_ratio[i] = 0;
	    	//cout << detection_ratio[i] << endl;
	    }

	    max_detection_ratio = distance(detection_ratio, max_element(detection_ratio, detection_ratio + NUM_OF_BOXES));
	    //sum_inliers_num = sum_element(inliers_int, inliers_int + NUM_OF_BOXES);
	    //cout << "Index of max element: " << max_detection_ratio << endl;

        string text = "Item " + IntToString(max_detection_ratio) + " found";
        string text5 = "Confidence:";
        //string text2 = "Inliers: " + inliers_str[i] + " - Outliers: " + outliers_str[i];

    	// if(detection_ratio[max_detection_ratio] != 100 && sum_inliers_num > 10 && sum_outliers_num > 10){
        if(true){
        	// drawText(frame_vis, text, blue);
        	drawText5(frame_vis, text5, blue);
        	drawConfidence1(frame_vis, detection_ratio[max_detection_ratio], yellow);
		}
		
		imshow("REAL TIME DEMO", frame_vis);
	
        if (!saveDirectory.empty())
        {
            const int widthSave = !frame_matching[max_detection_ratio].empty() ? frame_matching[max_detection_ratio].cols : frame_vis.cols;
            const int heightSave = !frame_matching[max_detection_ratio].empty() ? frame_matching[max_detection_ratio].rows + frame_vis.rows : frame_vis.rows;
            frameSave = Mat::zeros(heightSave, widthSave, CV_8UC3);
            if (!frame_matching[max_detection_ratio].empty())
            {
                int startX = (int)((widthSave - frame_vis.cols) / 2.0);
                Mat roi = frameSave(Rect(startX, 0, frame_vis.cols, frame_vis.rows));
                frame_vis.copyTo(roi);

                roi = frameSave(Rect(0, frame_vis.rows, frame_matching[max_detection_ratio].cols, frame_matching[max_detection_ratio].rows));
                frame_matching[max_detection_ratio].copyTo(roi);
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
