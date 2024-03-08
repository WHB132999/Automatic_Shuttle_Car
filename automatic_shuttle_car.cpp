#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <communication/multi_socket.h>
#include <models/tronis/ImageFrame.h>
#include <grabber/opencv_tools.hpp>
#include <models/tronis/BoxData.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <thread>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;
using namespace cv;

// Build the PID controller
class PidController
{
public:
    PidController( double Kp_, double Ki_, double Kd_ ) : Kp( Kp_ ), Ki( Ki_ ), Kd( Kd_ )
    {
        p_error = 0.;
        i_error = 0.;
        d_error = 0.;
    }

    /**
     * Update each term in the PID error variables given the current error
     * @param error The current error
     */
    inline void UpdateErrorTerms( double error )
    {
        d_error = error - p_error;
        i_error += error;
        p_error = error;
    }

    /**
     * Calculate the each term of the PID error
     * @param printValue print the error and result in pid controller
     * @output the total command to the actuator
     */
    double OutputToActuator( double d_limit, bool printValue )
    {
        /* optional limit on derivative term */
        if( Kd * d_error > d_limit )
            return Kp * p_error + Ki * i_error + d_limit;
        if( Kd * d_error < -d_limit )
            return Kp * p_error + Ki * i_error - d_limit;

        if( printValue )
        {
            printf( "p_error = %f, i_error = %f, d_error = %f \n", p_error, i_error, d_error );
            printf( "Kp *= %f, Ki *= %f, Kd *= %f \n", Kp * p_error, Ki * i_error, Kd * d_error );
        }

        return Kp * p_error + Ki * i_error + Kd * d_error;
    }

    /*set the i_error in pid controller to 0*/
    inline void setZero()
    {
        i_error = p_error = 0;
    }

private:
    /**
     * PID Error terms
     */
    double p_error;
    double i_error;
    double d_error;

    /**
     * PID Gain coefficients
     */
    double Kp;
    double Ki;
    double Kd;
};

// Class lane-assitant
class LaneAssistant
{
public:
    LaneAssistant() : pid_controller( 0.7, -0.000005, -0.0005 )
    {
    }

    bool processData( tronis::CircularMultiQueuedSocket& socket )
    {
        // do stuff with data
        socket.send( tronis::SocketData( "Ego Fahrzeug Geschwindigkeit >>> " + to_string( ego_velocity_ ) ) );
        // send results via socket
        // send steering value via socket
        getSteeringInput( socket );

        // send throttle value via socket
        getThrottleInput( socket);

        return true;
    }

protected:
    std::string image_name_;
    cv::Mat image_;
    tronis::LocationSub ego_location_;
    tronis::OrientationSub ego_orientation_;
    double ego_velocity_;

    // parameters in task 2
    Point ego_leftS, ego_leftE;
    Point ego_rightS, ego_rightE;
    Point directionS, directionE;
    double rows = 512, cols = 720;

    // parameters in task 3
    double steering_input;
    double steering_pc = 1.3; // left lane，1.3
    double steering_dc = -0.0005;
    double steering_ic = -0.1; // right lane，-0.1
    double Err_steering;
    double dErr_steering;
    double lastErr_steering = 0;
    double sumErr_steering;

    // parameters in task 4
    tronis::ObjectVector Objects_BBox;
    double throttle_input = 1;
    double throttle_pc = 0.5;
    double throttle_dc = -0.002;
    double throttle_ic = -0.02;
    double Err_velocity;
    double lastErr_velocity = 0;
    double dErr_velocity;
    double sumErr_velocity;

    PidController pid_controller;

    //*******************************************
    // Task 2: Lane detection
    vector<Vec4d> setLanes()
    {
        cv::Mat blur_img;
		// Reduce nosie
        GaussianBlur( image_, blur_img, Size( 3, 3 ), 0, 0, BORDER_DEFAULT );
        cv::Mat gray_img;
		// Transform to grey value image
        cvtColor( blur_img, gray_img, cv::COLOR_BGR2GRAY );
        cv::Mat binary_img;
		// Threshold the grey values
        cv::threshold( gray_img, binary_img, 120, 255, cv::THRESH_BINARY );
        cv::Mat edge_img;
		// Edge detection by Canny detector
        Canny( binary_img, edge_img, 100, 200 );

        // set polygon mask to only keep the region of interest
        cv::Mat mask = Mat::zeros( image_.size(), edge_img.type() );
        const int num = 6;
        Point points[1][num] = {
            Point( 0, rows * 0.85 ),           Point( 0, rows * 0.7 ),
            Point( cols * 0.33, rows * 0.65 ), Point( cols * 0.66, rows * 0.65 ),
            Point( cols, rows * 0.7 ),         Point( cols, rows * 0.85 )};
        /*point points[1][num] = {point( 0, rows ),
                                point( 0, rows * 0.7 ),
                                point( cols * 0.33, rows * 0.55 ),
                                point( cols * 0.66, rows * 0.55 ),
                                point( cols, rows * 0.7 ),
                                point( cols, rows )};*/
        const Point* polygon = points[0];
        fillConvexPoly( mask, polygon, num, Scalar( 255 ) );
        cv::Mat roi_img;
		// Using mask to get RoI
        cv::bitwise_and( edge_img, mask, roi_img );
        imshow( "Canny output: Region of Interest", edge_img );

        vector<Vec4d> raw_lanes;
		// Using Hough transformation to lines
        HoughLinesP( roi_img, raw_lanes, 1, CV_PI / 180, 50, 50, 10 );

        return raw_lanes;
    }

    // HSV_color
    vector<Vec4d> setWarnings_HSV()
    {
        cv::Mat blur_img;  // Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
        GaussianBlur( image_, blur_img, Size( 3, 3 ), 0, 0, BORDER_DEFAULT );

        // set the HSV range
        cv::Scalar scalarL = cv::Scalar( 0, 90, 90 );
        cv::Scalar scalarH = cv::Scalar( 22, 255, 255 );
        // H82 S185 V30			H129 S212 V37		H152 S155 V33

        cv::Mat hsv_img;
        cv::cvtColor( blur_img, hsv_img, COLOR_BGR2HSV );
		// Transform in HSV color range
        cv::inRange( hsv_img, scalarL, scalarH, hsv_img );
        //imshow( "HSV output: Region of Interest4", hsv_img );

        cv::Mat edge_img;  // Edge detection
        Canny( hsv_img, edge_img, 100, 200 );
        // imshow( "HSV output: Region of Interest5", edge_img );

        // set a polygon mask to only keep thed region of interest
        cv::Mat mask = Mat::zeros( image_.size(), edge_img.type() );
        const int num = 6;
        Point points[1][num] = {Point( 0, rows ),
                                Point( 0, rows * 0.7 ),
                                Point( cols * 0.33, rows * 0.55 ),
                                Point( cols * 0.66, rows * 0.55 ),
                                Point( cols, rows * 0.7 ),
                                Point( cols, rows )};
        const Point* polygon = points[0];
        fillConvexPoly( mask, polygon, num, Scalar( 255 ) );

        cv::Mat roi_img;
        cv::bitwise_and( edge_img, mask, roi_img );
        // imshow( "Canny output: Region of Interest6", roi_img );
        // imshow( "Canny output: Region of Interest7", edge_img );

        vector<Vec4d> warning_lanes;  // will hold all the results of the detection
        HoughLinesP( roi_img, warning_lanes, 1, CV_PI / 180, 50, 50,
                     10 );  // Probabilistic Line Transform

        return warning_lanes;
    }

    // BGR_color
    vector<Vec4d> setWarnings()
    {
        // balance of HSV (optional)
        vector<Mat> hsvSplit;  // vector to keep HSV 3 Channels info
        cv::Mat img_hsv;
        cv::Mat img_bgr;

        cvtColor( image_, img_hsv, COLOR_BGR2HSV );

        split( img_hsv, hsvSplit );                // classify the original image 3 HSV Channels
        equalizeHist( hsvSplit[2], hsvSplit[2] );  // balance the HSV light Channels
        merge( hsvSplit, img_hsv );                // merge all the channels
        cvtColor( img_hsv, img_bgr, COLOR_HSV2BGR );

        // classify the colors: the channels order should be BGR
        Mat img_threshold;

        inRange( img_bgr, Scalar( 0, 128, 128 ), Scalar( 127, 255, 255 ), img_threshold );
        /*
        blue:		(128,0,0) (255,127,127)
        white:		(128,128,128) (255,255,255)
        cyan:		(128,128,0) (255,255,127)
        purple:		(0,128,128) (127,255,255)
        yellow:		(0,128,128) (127,255,255)
        green:		(0,128,0) (127,255,127)
        red:		(0,0,128) (127,127,255)
        black:		(0,0,0,) (127,127,127)
        */
        // R154 G121 B20		R241 G224 B33		R222 G202 B101

        //// remove the noise
        // Mat element = getStructuringElement( MORPH_RECT, Size( 5, 5 ) );
        // morphologyEx( img_threshold, img_threshold, MORPH_OPEN, element );
        // morphologyEx( img_threshold, img_threshold, MORPH_CLOSE, element );

        // set a polygon mask to only keep thed region of interest
        cv::Mat mask = Mat::zeros( img_threshold.size(), img_threshold.type() );
        const int num = 6;
        Point points[1][num] = {Point( 0, rows ),
                                Point( 0, rows * 0.7 ),
                                Point( cols * 0.33, rows * 0.55 ),
                                Point( cols * 0.66, rows * 0.55 ),
                                Point( cols, rows * 0.7 ),
                                Point( cols, rows )};
        const Point* polygon = points[0];
        fillConvexPoly( mask, polygon, num, Scalar( 255 ) );

        // cv::bitwise_and( img_threshold, mask, img_threshold );
        imshow( "color", img_threshold );

        vector<Vec4d> warning_lanes;  // will hold all the results of the detection
        HoughLinesP( img_threshold, warning_lanes, 1, CV_PI / 180, 50, 50,
                     10 );  // Probabilistic Line Transform

        return warning_lanes;
    }

    void getLanes( vector<Vec4d> raw_lanes )
    {
        vector<Vec4d> left_lanes, right_lanes;
        Vec4f left_lane_function, right_lane_function;
        vector<Point> left_points, right_points;

        ego_leftS.y = 300;
        ego_rightS.y = 300;
        ego_leftE.y = 500;
        ego_rightE.y = 500;

        double left_k, right_k;
        Point left_b, right_b;

        for( auto lane : raw_lanes )
        {
            double lane_center = ( lane[0] + lane[2] ) / 2;
			// Smaller than width/2 -> left lane, otherwise right lane
            if( lane_center < cols / 2 )
            {
                left_lanes.push_back( lane );
            }
            else
            {
                right_lanes.push_back( lane );
            }
        }

		//for( auto line : raw_lanes )
  //      {
  //          double slope = ( line[3] - line[1] ) / ( line[2] - line[0] );  //(y2-y1)/(x2-x1)
  //          if( std::fabs( slope ) < 0.25 )  // only consider extreme slope
  //          {
  //              continue;
  //          }
  //          else if( slope < 0 )  // if the slope is negative, left group
  //          {
  //              left_lanes.push_back( line );
  //          }
  //          else
  //          {
  //              right_lanes.push_back( line );
  //          }
  //      }

        // get the left lines
        for( auto left_lane : left_lanes )
        {
            left_points.push_back( Point( left_lane[0], left_lane[1] ) );
            left_points.push_back( Point( left_lane[2], left_lane[3] ) );
        }
        if( left_points.size() > 0 )
        {
            // Using cv::fitLine to fit the left lane
			cv::fitLine( left_points, left_lane_function, cv::DIST_L2, 0, 0.01, 0.01 );
			left_k = left_lane_function[1] / left_lane_function[0];
            left_b = Point( left_lane_function[2], left_lane_function[3] );
            ego_leftS.x = ( ego_leftS.y - left_b.y ) / left_k + left_b.x;
            ego_leftE.x = ( ego_leftE.y - left_b.y ) / left_k + left_b.x;
        }
        // get the right lines
        for( auto right_lane : right_lanes )
        {
            right_points.push_back( Point( right_lane[0], right_lane[1] ) );
            right_points.push_back( Point( right_lane[2], right_lane[3] ) );
        }
        if( right_points.size() > 0 )
        {
            cv::fitLine( right_points, right_lane_function, cv::DIST_L2, 0, 0.01, 0.01 );
            right_k = right_lane_function[1] / right_lane_function[0];
            right_b = Point( right_lane_function[2], right_lane_function[3] );
            ego_rightS.x = ( ego_rightS.y - right_b.y ) / right_k + right_b.x;
            ego_rightE.x = ( ego_rightE.y - right_b.y ) / right_k + right_b.x;
        }
        if( ego_rightS.x > 0 && ego_leftS.x > 0 )
        {
            directionS = ( ego_leftS + ego_rightS ) / 2;
            directionE = ( ego_leftE + ego_rightE ) / 2;
        }
        else if( ego_rightS.x > 0 && ego_leftS.x <= 0 )
        {
            double a = 100;
            double c =
                std::atan( ( ego_rightE.x - ego_rightS.x ) / ( ego_rightE.y - ego_rightS.y ) ) -
                0.953;
            double b = ego_rightE.x - ego_rightS.x - ( ego_leftE.y - ego_leftS.y ) * tan( c );
            directionS.x = ego_rightS.x - a;
            directionE.x = ego_rightE.x - a - b;

            directionS.y = ego_rightS.y;
            directionE.y = ego_rightE.y;
        }
        else if( ego_rightS.x <= 0 && ego_leftS.x > 0 )
        {
            double a = 100;
            double c =
                std::atan( ( ego_leftS.x - ego_leftE.x ) / ( ego_leftE.y - ego_leftS.y ) ) - 0.953;
            double b = ego_leftS.x - ego_leftE.x - ( ego_leftE.y - ego_leftS.y ) * tan( c );
            directionS.x = ego_leftS.x + a;
            directionE.x = ego_leftE.x + a + b;

            directionS.y = ego_leftS.y;
            directionE.y = ego_leftE.y;
        }

        // Aufgabe3
        //directionS = ( ego_leftS + ego_rightS ) / 2;
        //directionE = ( ego_leftE + ego_rightE ) / 2;
    }
    //*******************************************

    // Function to detect lanes based on camera image
    void detectLanes( tronis::CircularMultiQueuedSocket& socket )
    {   // Detect white lanes or yellow lanes, white for normal driving, yellow for dangerous
        vector<Vec4d> white_lanes = setLanes();
        vector<Vec4d> yellow_lanes = setWarnings_HSV();
        std::cout << "white!" << white_lanes.size() << std::endl;
        std::cout << "yellow!" << yellow_lanes.size() << std::endl;
        if( yellow_lanes.size() > 0 && white_lanes.size() == 0)
        {
			std::cout << "Detect Yellow Lines!" << std::endl;
            getLanes( yellow_lanes );
		}
        /*else if( yellow_lanes.size() > 0 && white_lanes.size() > 0 )
        {
            getLanes( yellow_lanes );
        }*/
		else
		{       
			if (yellow_lanes.size() > 0 && white_lanes.size() > 0)
			{
                            throttle_input = -0.5;
                          std::cout << "throttle!" << throttle_input << std::endl;
                        socket.send( tronis::SocketData( " Throttle Value >>>" + to_string( throttle_input ) ) );
                            //setthrottleinput(-1.0);
                            //exit( 0 );
			}
			std::cout << "Only White lines!" << std::endl;
                    getLanes( white_lanes );
		}
        //exit(0);
        //getLanes( detect_lanes );

        // draw the lane lines and show the results
        line( image_, ego_leftS, ego_leftE, Scalar( 0, 0, 225 ), 3, LINE_AA );
        line( image_, ego_rightS, ego_rightE, Scalar( 0, 0, 225 ), 3, LINE_AA );

        // Task 3: Draw the driving direction lines and show results
        line( image_, Point( directionS.x, directionS.y ), Point( directionE.x, directionE.y ),
              Scalar( 0, 255, 0 ), 3, LINE_AA );
    }

    //*********************************************************
    // Aufgabe3: Steering control

    void setSteeringInput()
    {
        // Straight driving
		if( directionS.x == directionE.x )  // when the car drives straight
        {
            steering_input = 0;
        }
        else
		// Computing the slope of truning path
        {
            double slope =
                -( directionS.y - directionE.y ) /
                ( directionS.x - directionE.x );  // positive:	up right to down left
            double steering_winkel =
                M_PI_2 - abs( atan( slope ) );  // 0: vertical		pi/2:	horizontal

            // Computing steering_winkel
			Err_steering = steering_winkel / M_PI_2 - 0;
            dErr_steering = Err_steering - lastErr_steering;
            sumErr_steering = Err_steering + lastErr_steering;
            lastErr_steering = Err_steering;

            // pid_controller.UpdateErrorTerms( Err_steering );
            // steering_input = pid_controller.OutputToActuator( 0.5, true);

            steering_input = steering_pc * abs( Err_steering ) + steering_dc * dErr_steering + steering_ic * sumErr_steering;

			if( slope > 0 )  // drving to the right is positive
            {
                steering_input = -( steering_input );
            }
        }
    }

    void getSteeringInput( tronis::CircularMultiQueuedSocket& socket )
    {
        setSteeringInput();
        string prefix_steering = "Steering value >>> ";
        socket.send( tronis::SocketData( prefix_steering + to_string( steering_input ) ) );
    }

    //*************************************
    // Task 4: Throttle control

	// Get location, pose and velocity info of the driving car
    bool processPoseVelocity( tronis::PoseVelocitySub* msg )
    {
        ego_location_ = msg->Location;
        ego_orientation_ = msg->Orientation;
        ego_velocity_ = msg->Velocity * 3.6 * 1e-2; // from cm/s to Km/h
        return true;
    }
	// Object detection
    bool processObject( tronis::BoxDataSub* sensorData )
    {
        Objects_BBox = sensorData->Objects;
        return true;
    }
	// Computing the distance between the car and the detected objects
    double processDistance( tronis::LocationSub location )
    {
        float pos_x = location.X / 100;//cm to m
        float pos_y = location.Y / 100;
        double dist = sqrt( pow( pos_x, 2 ) + pow( pos_y, 2 ) );
        return dist;
    }

    void setThrottleInput( double dist )
    {
        Err_velocity = 60 - ego_velocity_;
        sumErr_velocity = lastErr_velocity + Err_velocity;
        dErr_velocity = lastErr_velocity - Err_velocity;
        lastErr_velocity = Err_velocity;

        //double min_distance = 15;
        double min_distance;
        if( ego_velocity_ < 50 )
        {
            min_distance = 15;
        }
		else
		{
                    min_distance = 0.5 * ego_velocity_;
		}
        std::cout << "Dist!" << dist << std::endl;
        std::cout << "Velocity!" << ego_velocity_ << std::endl;
        if( dist < min_distance )
        {
            if( abs( ego_velocity_ ) < 1 )
            {
                throttle_input = 0;
            }
            else
            {
                throttle_input = -1;
            }
        }
        else if( dist < min_distance + 5 && ego_velocity_ > 40 ||
                 dist < min_distance + 10 && ego_velocity_ > 45 )
        {
            throttle_input = -0.5;
        }
        else if( abs( Err_steering ) > 0.08 && ego_velocity_ > 45)
            {
            throttle_input = -1;
            }
        else if( abs( Err_steering ) > 0.03 && ego_velocity_ > 45 )
        {
            throttle_input = -0.5;
        }
        else
        {
			if (abs(ego_velocity_) > 20)
			{
                            throttle_input = throttle_pc * Err_velocity +
                                             throttle_dc * dErr_velocity +
                                             throttle_ic * sumErr_velocity;
			}
			else
			{
                            throttle_input = 1;
			}
        }
        if( throttle_input > 1 )
        {
            throttle_input = 1;
        }
		else if (throttle_input < -1)
		{
                    throttle_input = -1;
		}
		}

	//void getThrottleInput(tronis::CircularMultiQueuedSocket& socket)
	//{
 //           string prefix_throttle = "Throttle Value >>> ";
	//		if (Objects_BBox.size())
	//		{
	//			for (size_t i = 0; i < Objects_BBox.size(); i++)
	//			{                
	//				tronis::ObjectSub& object = Objects_BBox[i];
 //                                   cout << "Name Value!" << object.ActorName.Value() << endl;
 //                                       if( object.ActorName.Value() == "GenericCoupe_8" ||
 //                                           "Generic_Van_2" )  // GenericCoupe_8, Generic_Cabriolet_2,
 //                                                            // prp_trafficLight_Blueprint_21
	//								{
 //                                   double dist = processDistance(object.Pose.Location );
 //                                   setThrottleInput(dist );
 //                                   socket.send( tronis::SocketData(prefix_throttle + to_string(throttle_input ) ) );

	//								}
 //                                                                       /*else if( object.ActorName
 //                                                                                    .Value() ==
 //                                                                                "Generic_Van_2" )
 //                                                                       {
 //                                                                           double dist =
 //                                                                               processDistance(
 //                                                                                   object.Pose
 //                                                                                       .Location );
 //                                                                           setThrottleInput(
 //                                                                               dist );
 //                                                                           socket.send( tronis::SocketData(
 //                                                                               prefix_throttle +
 //                                                                               to_string(
 //                                                                                   throttle_input ) ) );
 //                                                                       }*/
	//								else
	//								{
 //                                   setThrottleInput(100.0 );
 //                                   socket.send( tronis::SocketData(prefix_throttle + to_string( throttle_input ) ) );
 //                                   cout << "Lane or Track got detected, please ignore!" << endl;
	//								}
	//						}
	//		}
	//		else
 //                       {
 //                           /*if( flag == true )
 //                           {
 //                               setThrottleInput( -1.0 );
 //                               socket.send( tronis::SocketData( prefix_throttle +
 //                                                                to_string( throttle_input ) ) );
 //                               cout << "Lowest Throttle!" << endl;
	//						}*/
 //                           setThrottleInput( 100.0 );
 //                           socket.send( tronis::SocketData( prefix_throttle + to_string( throttle_input ) ) );
 //                           cout << "No object in the Front!" << endl;
 //                           cout << "Right!" << endl;
	//		}
	//}
  //  void getThrottleInput( tronis::CircularMultiQueuedSocket& socket )
  //  {
  //      std::vector<double> distList;
		//string prefix_throttle = "Throttle Value >>> ";
  //      if( Objects_BBox.size() )
  //      {
  //          for( size_t i = 0; i < Objects_BBox.size(); i++ )
  //          {
  //              tronis::ObjectSub& object = Objects_BBox[i];
  //              cout << "Name Value!" << object.ActorName.Value() << endl;
  //              if( object.ActorName.Value() == "GenericCoupe_8" ||
  //                  "Generic_Van_2" )  // GenericCoupe_8, Generic_Cabriolet_2,
  //                                     // prp_trafficLight_Blueprint_21
  //              {
  //                  double dist = processDistance( object.Pose.Location );
  //                  distList.push_back( dist );
  //              }
  //              /*else if( object.ActorName
  //                           .Value() ==
  //                       "Generic_Van_2" )
  //              {
  //                  double dist =
  //                      processDistance(
  //                          object.Pose
  //                              .Location );
  //                  setThrottleInput(
  //                      dist );
  //                  socket.send( tronis::SocketData(
  //                      prefix_throttle +
  //                      to_string(
  //                          throttle_input ) ) );
  //              }*/
  //              else
  //              {
  //                  setThrottleInput( 100.0 );
  //                  socket.send(
  //                      tronis::SocketData( prefix_throttle + to_string( throttle_input ) ) );
  //                  cout << "Lane or Track got detected, please ignore!" << endl;
  //              }
  //          }
  //      }
  //      else
  //      {
  //          /*if( flag == true )
  //          {
  //              setThrottleInput( -1.0 );
  //              socket.send( tronis::SocketData( prefix_throttle +
  //                                               to_string( throttle_input ) ) );
  //              cout << "Lowest Throttle!" << endl;
  //                                      }*/
  //          setThrottleInput( 100.0 );
  //          socket.send( tronis::SocketData( prefix_throttle + to_string( throttle_input ) ) );
  //          cout << "No object in the Front!" << endl;
  //          cout << "Right!" << endl;
  //      }
  //  }
    //void getThrottleInput( tronis::CircularMultiQueuedSocket& socket )
    //{
    //    string prefix_throttle = "Throttle value >>> ";
    //    double minDist;
    //    std::vector<double> newList;
    //    for( size_t i = 0; i < Objects_BBox.size(); i++ )
    //    {
    //        tronis::ObjectSub& object = Objects_BBox[i];

    //        cout << "value of actor name: " << object.ActorName.Value() << endl;
    //        double dist = processDistance( object.Pose.Location );
    //        cout << "Dists: " << dist << endl;
    //        //cout << "Objects!!!: " << typeid(object.ActorName.Value()).name() << endl;

    //        if( object.ActorName.Value() == "prp_trafficLight_Blueprint_21" ||
    //            object.ActorName.Value() == "GenericCoupe_8" )
    //        {
    //            double dist = processDistance( object.Pose.Location );

    //            newList.push_back( dist );
    //        }
    //    }

    //    for( size_t i = 0; i < newList.size(); i++ )
    //    {
    //        cout << "dist in the new list: " << newList[i] << endl;
    //    }
    //    auto minElementIterator = std::min_element( newList.begin(), newList.end() );
    //    if( minElementIterator != newList.end() )
    //    {
    //        minDist = *minElementIterator;
    //        cout << "minDist: " << minDist << endl;
    //        setThrottleInput( minDist );
    //        socket.send( tronis::SocketData( prefix_throttle + to_string( throttle_input ) ) );
    //    }
    //    else
    //    {
    //        cout << "no  objects are detected" << endl;
    //        setThrottleInput( 100.0 );
    //        socket.send(
    //            tronis::SocketData( prefix_throttle + to_string( throttle_input ) ) );  ///???
    //        cout << "no objects in the front !" << endl;
    //    }
    //}
	void getThrottleInput( tronis::CircularMultiQueuedSocket& socket )
    {
        string prefix_throttle = "Throttle value >>> ";
        double minDist;
        std::vector<double> newList;
        bool waitAndReset = false;  // Flag to indicate whether we need to wait and reset

        for( size_t i = 0; i < Objects_BBox.size(); i++ )
        {
            tronis::ObjectSub& object = Objects_BBox[i];

            cout << "value of actor name: " << object.ActorName.Value() << endl;

            if( object.ActorName.Value() == "GenericCoupe_8" ||
                object.ActorName.Value() == "prp_trafficLight_Blueprint_21" )
            {
                double dist = processDistance( object.Pose.Location );

                newList.push_back( dist );
            }
        }

        for( size_t i = 0; i < newList.size(); i++ )
        {
            cout << "dist in the new list: " << newList[i] << endl;
        }

        auto minElementIterator = std::min_element( newList.begin(), newList.end() );
        if( minElementIterator != newList.end() )
        {
            minDist = *minElementIterator;
            cout << "minDist: " << minDist << endl;

            // Check if the object is "prp_trafficLight_Blueprint_2"
            tronis::ObjectSub& object =
                Objects_BBox[std::distance( newList.begin(), minElementIterator )];
            cout << "Object!!!: " << object.ActorName.Value() << endl;
			if( (object.ActorName.Value() == "BP_SnappyRoad2_5") &&
                (ego_velocity_ < 0.00001) )
            {
                // Set the flag to true if "prp_trafficLight_Blueprint_2" is detected
                waitAndReset = true;
                cout << "wait and reset!!!" << endl;
            }

            setThrottleInput( minDist );
            socket.send( tronis::SocketData( prefix_throttle + to_string( throttle_input ) ) );
        }
        else
        {
            cout << "no objects are detected" << endl;
            setThrottleInput( 100.0 );
            socket.send( tronis::SocketData( prefix_throttle + to_string( throttle_input ) ) );
            
        }

        if( waitAndReset )
        {
            // Wait for 5 seconds before resetting the throttle input
            std::this_thread::sleep_for( std::chrono::seconds( 6 ) );
            setThrottleInput( 100.0 );
            
            socket.send( tronis::SocketData( prefix_throttle + to_string( throttle_input ) ) );
            cout << "Resetting throttle input after waiting!" << endl;
            std::this_thread::sleep_for( std::chrono::seconds( 2 ) );
        }
    }


public:
    // Function to process received tronis data
    bool getData( tronis::ModelDataWrapper data_model, tronis::CircularMultiQueuedSocket& socket )
    {
        if( data_model->GetModelType() == tronis::ModelType::Tronis )
        {
            std::cout << "Id: " << data_model->GetTypeId() << ", Name: " << data_model->GetName()
                      << ", Time: " << data_model->GetTime() << std::endl;

            // if data is sensor output, process data
            switch( static_cast<tronis::TronisDataType>( data_model->GetDataTypeId() ) )
            {
                case tronis::TronisDataType::Image:
                {
                    processImage( data_model->GetName(),
                                  data_model.get_typed<tronis::ImageSub>()->Image, socket );
                    break;
                }
                case tronis::TronisDataType::ImageFrame:
                {
                    const tronis::ImageFrame& frames(
                        data_model.get_typed<tronis::ImageFrameSub>()->Images );
                    for( size_t i = 0; i != frames.numImages(); ++i )
                    {
                        std::ostringstream os;
                        os << data_model->GetName() << "_" << i + 1;

                        processImage( os.str(), frames.image( i ) , socket);
                    }
                    break;
                }
                case tronis::TronisDataType::ImageFramePose:
                {
                    const tronis::ImageFrame& frames(
                        data_model.get_typed<tronis::ImageFramePoseSub>()->Images );
                    for( size_t i = 0; i != frames.numImages(); ++i )
                    {
                        std::ostringstream os;
                        os << data_model->GetName() << "_" << i + 1;

                        processImage( os.str(), frames.image( i ) , socket);
                    }
                    break;
                }
                case tronis::TronisDataType::PoseVelocity:
                {
                    processPoseVelocity( data_model.get_typed<tronis::PoseVelocitySub>() );
                    break;
                }
                /*case tronis::tronisdatatype::object:
                {
                    processobject();
                    break;
                }*/
                case tronis::TronisDataType::BoxData:
				{
                    processObject( data_model.get_typed<tronis::BoxDataSub>() );
					//std::cout << data_model.get_typed<tronis::BoxDataSub>()->ToString() << std::endl;
                    break;
				}
                default:
                {
                    std::cout << data_model->ToString() << std::endl;
                    break;
                }
            }
            return true;
        }
        else
        {
            std::cout << data_model->ToString() << std::endl;
            return false;
        }
    }

protected:
    // Function to show an openCV image in a separate window
    void showImage( std::string image_name, cv::Mat image )
    {
        cv::Mat out = image;
        if( image.type() == CV_32F || image.type() == CV_64F )
        {
            cv::normalize( image, out, 0.0, 1.0, cv::NORM_MINMAX, image.type() );
        }
        cv::namedWindow( image_name.c_str(), cv::WINDOW_NORMAL );
        cv::imshow( image_name.c_str(), out );
    }

    // Function to convert tronis image to openCV image
    bool processImage( const std::string& base_name, const tronis::Image& image,
                       tronis::CircularMultiQueuedSocket& socket )
    {
        std::cout << "processImage" << std::endl;
        if( image.empty() )
        {
            std::cout << "empty image" << std::endl;
            return false;
        }

        image_name_ = base_name;
        image_ = tronis::image2Mat( image );

        detectLanes(socket);
        showImage( image_name_, image_ );

        return true;
    }
};

// main loop opens socket and listens for incoming data
int main( int argc, char** argv )
{
    std::cout << "Welcome to lane assistant" << std::endl;

    // specify socket parameters
    std::string socket_type = "TcpSocket";
    std::string socket_ip = "127.0.0.1";
    std::string socket_port = "7778";

    std::ostringstream socket_params;
    socket_params << "{Socket:\"" << socket_type << "\", IpBind:\"" << socket_ip
                  << "\", PortBind:" << socket_port << "}";

    int key_press = 0;  // close app on key press 'q'
    tronis::CircularMultiQueuedSocket msg_grabber;
    //tronis::CircularMultiQueuedSocket socket;
    uint32_t timeout_ms = 500;  // close grabber, if last received msg is older than this param

    LaneAssistant lane_assistant;

    while( key_press != 'q' )
    {
        std::cout << "Wait for connection..." << std::endl;
        msg_grabber.open_str( socket_params.str() );

        if( !msg_grabber.isOpen() )
        {
            printf( "Failed to open grabber, retry...!\n" );
            continue;
        }

        std::cout << "Start grabbing" << std::endl;
        tronis::SocketData received_data;
        uint32_t time_ms = 0;

        while( key_press != 'q' )
        {
            // wait for data, close after timeout_ms without new data
            if( msg_grabber.tryPop( received_data, true ) )
            {
                // data received! reset timer
                time_ms = 0;

                // convert socket data to tronis model data
                tronis::SocketDataStream data_stream( received_data );
                tronis::ModelDataWrapper data_model(
                    tronis::Models::Create( data_stream, tronis::MessageFormat::raw ) );
                if( !data_model.is_valid() )
                {
                    std::cout << "received invalid data, continue..." << std::endl;
                    continue;
                }
                // identify data type
                lane_assistant.getData( data_model, msg_grabber );
                lane_assistant.processData( msg_grabber );
            }
            else
            {
                // no data received, update timer
                ++time_ms;
                if( time_ms > timeout_ms )
                {
                    std::cout << "Timeout, no data" << std::endl;
                    msg_grabber.close();
                    break;
                }
                else
                {
                    std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
                    key_press = cv::waitKey( 1 );
                }
            }
        }
        msg_grabber.close();
    }
    return 0;
}
