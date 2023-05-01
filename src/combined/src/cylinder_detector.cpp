#include <iostream>
#include <ros/ros.h>
#include <math.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include "pcl/point_cloud.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "geometry_msgs/PointStamped.h"
#include <vector>
#include "geometry_msgs/Pose.h"
#include <string>
#include <visualization_msgs/MarkerArray.h>

#include <pcl/filters/voxel_grid.h>

#include <sensor_msgs/LaserScan.h>
#include <laser_geometry/laser_geometry.h>

// for custom messages about greeting the detected cylinder
#include "combined/CylinderGreetInstructions.h"

// for laser projection
//laser_geometry::LaserProjection projector_;
//tf::TransformListener listener_


// for memory management check
#include <memory>
/*
struct AllocationMetrics {
    uint32_t TotalAllocated = 0;
    uint32_t TotalFreed = 0;
    uint32_t CurrentUsage() {return (TotalAllocated - TotalFreed);}
    //uint32_t CurrentUsage() {return (TotalAllocated - TotalFreed) / 1000;}
};

static AllocationMetrics s_AllocationMetrics;

void* operator new(size_t size) {
    s_AllocationMetrics.TotalAllocated += size;

    return malloc(size);
}

void operator delete(void* memory, size_t size) {
    s_AllocationMetrics.TotalFreed += size;

    free(memory);
}

static void printMemoryUsage() {
    std::cout << "Memory usage: " << s_AllocationMetrics.CurrentUsage() << " bytes\n";
}
*/

// for publishing greet instructions of detected cylinders to brain
ros::Publisher greet_publisher;

ros::Publisher pubx;
ros::Publisher puby;
ros::Publisher pubm;
ros::Publisher cilinder_arr_pub;

tf2_ros::Buffer tf2_buffer;

typedef pcl::PointXYZRGB PointT;

//typedef geometry_msgs::Pose Pose;

// for holding cylinder detection variables

// number of votes for object position
std::vector < int > object_position_votes;
// current coordinates of detection
std::vector < geometry_msgs::Pose > potential_object_positions;
// true if position was already sent to brain
std::vector < bool > objects_already_sent;
// color of object at position i
std::vector < std::string > detected_object_colors;
// colors
std::vector < double > object_color_red;
std::vector < double > object_color_green;
std::vector < double > object_color_blue;

// how close the detections have to be to be considered the same object
double object_proximity_threshold = 1;
// how many detections before sending
//double object_num_detections_threshold = 2;
double object_num_detections_threshold = 5;

int marker_id = 0;

ros::Publisher brain_publisher;

visualization_msgs::MarkerArray cylinder_marker_arr;

double distance_euclidean(double x1, double y1, double x2, double y2) {
    return sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2));
}

double distance_euclidean_pose(geometry_msgs::Pose pose1, geometry_msgs::Pose pose2) {
    return distance_euclidean(pose1.position.x, pose1.position.y, pose2.position.x, pose2.position.y);
}

/*
Returns array of [r, g, b, a] that corresponds to string color
*/
std::vector<double> get_color_from_string(std::string color_str) {
    double r = 255;
    double g = 255;
    double b = 255;
    double a = 0.9;
    // for unknown color we will use white

    
    if (color_str == "red") {
        //res = [1, 0, 0, 1];
        r = 1;
        g = 0;
        b = 0;
        a = 1;
    }

    if (color_str == "green") {
        //res = [0, 1, 0, 1];
        r = 0;
        g = 1;
        b = 0;
        a = 1;
    }

    if (color_str == "blue") {
        //res = [0, 0, 1, 1];
        r = 0;
        g = 0;
        b = 1;
        a = 1;
    }

    if (color_str == "black") {
        //res = [0, 0, 0, 1];
        r = 0;
        g = 0;
        b = 0;
        a = 1;
    }

    if (color_str == "yellow") {
        //res = [255, 165, 0, 1];
        r = 255;
        g = 165;
        b = 0;
        a = 1;
    }
    

    std::vector<double> res{ r, g, b, a};
    return res;
    
}

/*
Does everything to deal with new confirmed object
*/
void publish_new_confirmed_object(geometry_msgs::Pose pose, std::string color_str) {
    /*
    // replace possible nans
    if(pose.position.x != pose.position.x) {
        pose.position.x = 0.0
    }

    if (pose.position.y != pose.position.y) {
        pose.position.y = 0.0
    }

    if (pose.position.z != pose.position.z) {
        pose.position.z = 0.0
    }
    */

    // get color for marker
    std::vector<double> marker_custom_color = get_color_from_string(color_str);

    // publish it to rviz
    visualization_msgs::Marker marker;

    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();

    marker.ns = "cylinder";
    marker.id = marker_id;
    marker_id++;

    marker.type = visualization_msgs::Marker::CYLINDER;
    marker.action = visualization_msgs::Marker::ADD;

    marker.pose = pose;

    marker.scale.x = 0.1;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;

    /*
    marker.color.r = 0.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0f;
    */
    
    
    marker.color.r = marker_custom_color[0];
    marker.color.g = marker_custom_color[1];
    marker.color.b = marker_custom_color[2];
    marker.color.a = marker_custom_color[3];
    

    marker.lifetime = ros::Duration();

    cylinder_marker_arr.markers.push_back(marker);

    //pubm.publish(marker);
    cilinder_arr_pub.publish(cylinder_marker_arr);
    std::cout << "Publishing unique cylinder with color: " << color_str << std::endl;

    // publish greet instructions to brain
    combined::CylinderGreetInstructions msg;
    msg.object_pose = pose;
    msg.object_id = marker_id - 1;
    msg.object_color = color_str;
    greet_publisher.publish(msg);
}

std::string
get_color(double red, double green, double blue){
	if (red > green && red > blue){
		//Probably red
		//if(abs(red-green) < 20){
        if(abs(red-green) < 15){
			//Probably yellow
			return "yellow";
		}
		
		return "red";
	}else if (green > red && green > blue){
		//Probably green
		//if(abs(red-green) < 20){
        if(abs(red-green) < 15){
			//Probably yellow
			return "yellow";
		}
		
		return "green";
	}else if (blue > red && blue > green){
		//Probably blue
		return "blue";
	}
	
	return "unknown";
}

/*
Deals with new cylinder pose
*/
void new_potential_object(geometry_msgs::Pose pose, double red, double green, double blue) {
    /*
    // if there are nans, we do not use them
    if(pose.position.x != pose.position.x) {
        return
    }

    if (pose.position.y != pose.position.y) {
        return
    }

    if (pose.position.z != pose.position.z) {
        return
    }
    */

    bool found_close_instance = false;
    // find closest object if available
    for (int i = 0; i < potential_object_positions.size(); i++) {
        std::cout << "Psose x: " << pose.position.x << " " << std::endl;
        std::cout << "Euclidean: " << distance_euclidean_pose(potential_object_positions[i], pose) << " " << std::endl;
        if (distance_euclidean_pose(potential_object_positions[i], pose) < object_proximity_threshold) {

            // update coordinates with weighted average
            potential_object_positions[i].position.x = (potential_object_positions[i].position.x * object_position_votes[i] + pose.position.x) / (object_position_votes[i] + 1);
            potential_object_positions[i].position.y = (potential_object_positions[i].position.y * object_position_votes[i] + pose.position.y) / (object_position_votes[i] + 1);
            potential_object_positions[i].position.z = (potential_object_positions[i].position.z * object_position_votes[i] + pose.position.z) / (object_position_votes[i] + 1);

	    object_color_red[i] = (object_color_red[i] * object_position_votes[i] + red) / (object_position_votes[i] + 1);
	    object_color_green[i] = (object_color_green[i] * object_position_votes[i] + green) / (object_position_votes[i] + 1);
	    object_color_blue[i] = (object_color_blue[i] * object_position_votes[i] + blue) / (object_position_votes[i] + 1);
	    
            // vote for current face
            object_position_votes[i] = object_position_votes[i] + 1;
            found_close_instance = true;

            // check if enough instances to be considered an object
            if ((object_position_votes[i] >= object_num_detections_threshold) && (objects_already_sent[i] == false)) {
                //if ((object_position_votes[i] > object_num_detections_threshold)) {
		std::string color = get_color(object_color_red[i], object_color_green[i], object_color_blue[i]);
                publish_new_confirmed_object(potential_object_positions[i], color);
                objects_already_sent[i] = true;

                std::cout << "New one published AAAAAAAAAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBBBBBBBBBBBBBCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC." << std::endl;
            }
        }
    }
    
    int n = object_position_votes.size();
    for (int iii = 0; iii < n; iii++) {
    	if(iii == n-1){
    		std::cout << object_position_votes[iii] << std::endl;
    		break;
    	}
    	
        std::cout << object_position_votes[iii] << ", ";
    }

    // if no closest found just append it
    if (!found_close_instance) {
        potential_object_positions.push_back(pose);
        object_position_votes.push_back(1);
        objects_already_sent.push_back(false);
        detected_object_colors.push_back("Unknown");
	    
	object_color_red.push_back(red);
	object_color_green.push_back(green);
	object_color_blue.push_back(blue);
    }
}

//////////////////////////////////// alternative cloud cb start

// only keep points based on conditions defined in params
void do_pass_through_filter(const pcl::PointCloud<PointT>::Ptr& cloud) {
    //pcl::PointCloud < PointT > ::Ptr cloud_filtered(new pcl::PointCloud < PointT > );

    pcl::PassThrough < PointT > pass;
    //pcl::PassThrough<pcl::PointXYZRGB> pass;

    pass.setInputCloud(cloud);
    //pass.setFilterFieldName("z");
    pass.setFilterFieldName("y");
    //pass.setFilterLimits(0, 1);
    // -1.1,0.1
    pass.setFilterLimits(-0.3, 0.2);
    //pass.setFilterLimits(0.1, 0.2);
    pass.filter( * cloud);
    //pass.filter( * cloud_filtered);
    //*cloud = *cloud_filtered;
}

void get_normals_for_cloud(const pcl::PointCloud<PointT>::Ptr& cloud,
                      const pcl::PointCloud<pcl::Normal>::Ptr& cloud_normals) {
    pcl::search::KdTree < PointT > ::Ptr tree(new pcl::search::KdTree < PointT > ());
    pcl::NormalEstimation < PointT, pcl::Normal > ne;

    // Estimate point normals
    ne.setSearchMethod(tree);
    ne.setInputCloud(cloud);
    ne.setKSearch(50);
    ne.compute( * cloud_normals);
}

/*
Downsamples pointcloud and converts it
*/
void downsample_pointcloud(const pcl::PCLPointCloud2ConstPtr & cloud_blob, 
                        const pcl::PointCloud<PointT>::Ptr& cloud) {
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    pcl::PCLPointCloud2::Ptr cloud_filtered_blob (new pcl::PCLPointCloud2);
    sor.setInputCloud (cloud_blob);
    sor.setLeafSize (0.01f, 0.01f, 0.01f);
    //sor.setLeafSize (0.005f, 0.005f, 0.005f);
    sor.filter (*cloud_filtered_blob);
    pcl::fromPCLPointCloud2( * cloud_filtered_blob, * cloud);
}

/*
remove plane from @param cloud
*/
void remove_plane(const pcl::PointCloud<PointT>::Ptr& cloud,
                    const pcl::PointIndices::Ptr& inliers_plane) {
    //
    // create a SAC segmenter without using normals
    pcl::SACSegmentation<PointT> segmentor;
    segmentor.setOptimizeCoefficients(true);
    segmentor.setModelType(pcl::SACMODEL_PLANE);
    segmentor.setMethodType(pcl::SAC_RANSAC);
    /* run at max 1000 iterations before giving up */
    segmentor.setMaxIterations(1000);
    /* tolerance for variation from model */
    segmentor.setDistanceThreshold(0.01);
    segmentor.setInputCloud(cloud);
    /* Create the segmentation object for the planar model and set all the parameters */
    pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients);
    segmentor.segment(*inliers_plane, *coefficients_plane);
    /* Extract the planar inliers from the input cloud */
    pcl::ExtractIndices<PointT> extract_indices;
    extract_indices.setInputCloud(cloud);
    extract_indices.setIndices(inliers_plane);
    /* Remove the planar inliers, extract the rest */
    extract_indices.setNegative(true);
    extract_indices.filter(*cloud);
}

void extract_normals(const pcl::PointCloud<pcl::Normal>::Ptr& cloud_normals,
                      const pcl::PointIndices::Ptr& inliers_plane) {

    pcl::ExtractIndices<pcl::Normal> extract_normals;
    extract_normals.setNegative(true);
    extract_normals.setInputCloud(cloud_normals);
    extract_normals.setIndices(inliers_plane);
    extract_normals.filter(*cloud_normals);
}

void get_cylinder (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
                       const pcl::ModelCoefficients::Ptr& coefficients_cylinder,
                       const pcl::PointCloud<pcl::Normal>::Ptr& cloud_normals) {
    return;
}



///////////////////////////////////// with depth start
void 
cloud_cb_basic_depth (const pcl::PCLPointCloud2ConstPtr& cloud_blob)
{
    // memory test
    //printMemoryUsage();

    // All the objects needed

    ros::Time time_rec, time_test;
    time_rec = ros::Time::now();
    
    pcl::PassThrough<PointT> pass;
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg; 
    pcl::PCDWriter writer;
    pcl::ExtractIndices<PointT> extract;
    pcl::ExtractIndices<pcl::Normal> extract_normals;
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
    Eigen::Vector4f centroid;

    // Datasets
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<PointT>::Ptr cloud_filtered2 (new pcl::PointCloud<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2 (new pcl::PointCloud<pcl::Normal>);
    pcl::ModelCoefficients::Ptr coefficients_plane (new pcl::ModelCoefficients), coefficients_cylinder (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices), inliers_cylinder (new pcl::PointIndices);

    pcl::PointCloud<PointT>::Ptr cloud_filtered_depth (new pcl::PointCloud<PointT>);
    
    // Read in the cloud data
    //pcl::fromPCLPointCloud2 (*cloud_blob, *cloud);

    // downsample cloud data
    downsample_pointcloud(cloud_blob, cloud);

    std::cout << "PointCloud has: " << cloud->points.size () << " data points." << std::endl;

    // if cloud is empty 
    /*
    if (cloud->points.size() <= 0) {
        return;
    }
    */

    // do passthrough over z
    pass.setInputCloud(cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0, 1.9);
    pass.filter (*cloud_filtered_depth);

    // Build a passthrough filter to remove spurious NaNs
    //pass.setInputCloud (cloud);
    pass.setInputCloud (cloud_filtered_depth);
    //pass.setFilterFieldName ("z");
    //pass.setFilterLimits (0, 1.5);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-0.3, 0.2);

    pass.filter (*cloud_filtered);
    std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size () << " data points." << std::endl;

    if (cloud_filtered->points.size() <= 10) {
        return;
    }

    // Estimate point normals
    ne.setSearchMethod (tree);
    ne.setInputCloud (cloud_filtered);
    ne.setKSearch (50);
    ne.compute (*cloud_normals);

    // Create the segmentation object for the planar model and set all the parameters
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_NORMAL_PLANE);
    seg.setNormalDistanceWeight (0.1);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (100);
    seg.setDistanceThreshold (0.03);
    seg.setInputCloud (cloud_filtered);
    seg.setInputNormals (cloud_normals);
    // Obtain the plane inliers and coefficients
    seg.segment (*inliers_plane, *coefficients_plane);
    std::cout << "Plane coefficients: " << *coefficients_plane << std::endl;

    // Extract the planar inliers from the input cloud
    extract.setInputCloud (cloud_filtered);
    extract.setIndices (inliers_plane);
    extract.setNegative (false);

    // Write the planar inliers to disk
    pcl::PointCloud<PointT>::Ptr cloud_plane (new pcl::PointCloud<PointT> ());
    extract.filter (*cloud_plane);
    std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

    // Remove the planar inliers, extract the rest
    extract.setNegative (true);
    extract.filter (*cloud_filtered2);
    extract_normals.setNegative (true);
    extract_normals.setInputCloud (cloud_normals);
    extract_normals.setIndices (inliers_plane);
    extract_normals.filter (*cloud_normals2);

    // if there are no more points left
    if (cloud_filtered2->points.size() <= 10) {
        return;
    }

    // Create the segmentation object for cylinder segmentation and set all the parameters
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_CYLINDER);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setNormalDistanceWeight (0.1);
    seg.setMaxIterations (10000);
    seg.setDistanceThreshold (0.05);
    //seg.setRadiusLimits (0.06, 0.2);
    seg.setRadiusLimits (0.11, 0.18);

    
    seg.setInputCloud (cloud_filtered2);
    seg.setInputNormals (cloud_normals2);
    
    /*
    seg.setInputCloud (cloud_filtered);
    seg.setInputNormals (cloud_normals);
    */


    // Obtain the cylinder inliers and coefficients
    seg.segment (*inliers_cylinder, *coefficients_cylinder);

    // if no model is found
    if (coefficients_cylinder->values.size() == 0) {
        std::cout << "No cylinder found mmm: "<< std::endl;
        return;
    }


    std::cout << "Cylinder coefficients: " << *coefficients_cylinder << std::endl;

    // Write the cylinder inliers to disk
    extract.setInputCloud (cloud_filtered2);
    extract.setIndices (inliers_cylinder);

    // check if valid
    if(coefficients_cylinder->values[6] < 0.1 || coefficients_cylinder->values[6] > 0.13) {
        return;
    }

    std::cout << "Inliers_num_cylinder: " << inliers_cylinder->indices.size() << std::endl;
    if (inliers_cylinder->indices.size() < 1200) {
        return;
    }

    extract.setNegative (false);
    pcl::PointCloud<PointT>::Ptr cloud_cylinder (new pcl::PointCloud<PointT> ());
    extract.filter (*cloud_cylinder);
    if (cloud_cylinder->points.empty ()) 
        std::cout << "Can't find the cylindrical component." << std::endl;
    else
    {
        std::cout << "PointCloud representing the cylindrical component: " << cloud_cylinder->points.size () << " data points." << std::endl;
            
            pcl::compute3DCentroid (*cloud_cylinder, centroid);
            std::cout << "centroid of the cylindrical component: " << centroid[0] << " " <<  centroid[1] << " " <<   centroid[2] << " " <<   centroid[3] << std::endl;

        //Create a point in the "camera_rgb_optical_frame"
            geometry_msgs::PointStamped point_camera;
            geometry_msgs::PointStamped point_map;
            visualization_msgs::Marker marker;
            geometry_msgs::TransformStamped tss;
            
            point_camera.header.frame_id = "camera_rgb_optical_frame";
            point_camera.header.stamp = ros::Time::now();

            point_map.header.frame_id = "map";
            point_map.header.stamp = ros::Time::now();

            point_camera.point.x = centroid[0];
            point_camera.point.y = centroid[1];
            point_camera.point.z = centroid[2];

        try{
            time_test = ros::Time::now();

            std::cout << time_rec << std::endl;
            std::cout << time_test << std::endl;
            tss = tf2_buffer.lookupTransform("map","camera_rgb_optical_frame", time_rec);
            //tf2_buffer.transform(point_camera, point_map, "map", ros::Duration(2));
        }
            catch (tf2::TransformException &ex)
        {
            ROS_WARN("Transform warning: %s\n", ex.what());
        }

            //std::cout << tss ;

            tf2::doTransform(point_camera, point_map, tss);

            std::cout << "point_camera: " << point_camera.point.x << " " <<  point_camera.point.y << " " <<  point_camera.point.z << std::endl;

            std::cout << "point_map: " << point_map.point.x << " " <<  point_map.point.y << " " <<  point_map.point.z << std::endl;

            // too high point for xylinder
            if (point_map.point.z > 0.75){
                std::cout << "Too high!" << std::endl;
                return;
            }

            // if on the floor
            if (point_map.point.z < 0.05) {
                std::cout << "Too low!" << std::endl;
                return;
            }
            
                   
        double red = 0;
        double green = 0;
        double blue = 0;
        
        for(int iii = 0; iii < cloud_cylinder -> points.size(); iii++){
            red = red + cloud_cylinder -> points[iii].r;
            green = green + cloud_cylinder -> points[iii].g;
            blue = blue +cloud_cylinder -> points[iii].b;
        }
        
        red = red / cloud_cylinder -> points.size();
        green = green / cloud_cylinder -> points.size();
        blue = blue / cloud_cylinder -> points.size();
        
        std::string color = get_color(red, green, blue);
        
        std::cout << "Red: " << red << ", Green: " << green << ", Blue:" << blue << ", Color: " << color << std::endl;

        geometry_msgs::Pose pose_new;

        pose_new.position.x = point_map.point.x;
        pose_new.position.y = point_map.point.y;
        pose_new.position.z = point_map.point.z;
        pose_new.orientation.x = 0.0;
        pose_new.orientation.y = 0.0;
        pose_new.orientation.z = 0.0;
        pose_new.orientation.w = 1.0;

        // if there are nans -> do not use them
        if(pose_new.position.x != pose_new.position.x) {
            return;
        }

        if (pose_new.position.y != pose_new.position.y) {
            return;
        }

        if (pose_new.position.z != pose_new.position.z) {
            return;
        }

        // declare new cylinder detection and render it
        new_potential_object(pose_new, red, green, blue);
        

        // for debug !!! TODO: remove after done testing
        /*
        pcl::PCLPointCloud2 outcloud_cylinder;
        pcl::toPCLPointCloud2 (*cloud_cylinder, outcloud_cylinder);
        puby.publish (outcloud_cylinder);
        */
        //publish_new_confirmed_object(pose_new, "test");

    }
  
}

int
main(int argc, char ** argv) {

    // Initialize ROS
    ros::init(argc, argv, "cylinder_segment");
    ros::NodeHandle nh;

    // For transforming between coordinate frames
    tf2_ros::TransformListener tf2_listener(tf2_buffer);

    // Create a ROS subscriber for the input point cloud
    //ros::Subscriber sub = nh.subscribe("/camera/depth/points", 1, cloud_cb);
    // test !!!
    //ros::Subscriber sub = nh.subscribe("/camera/depth/points", 1, cloud_cb1);
    // test !!!
    //ros::Subscriber sub = nh.subscribe("/camera/depth/points", 1, cloud_cb_basic);

    ros::Subscriber sub = nh.subscribe("/camera/depth/points", 1, cloud_cb_basic_depth);

    // Create a ROS publisher for the output point cloud
    //pubx = nh.advertise < pcl::PCLPointCloud2 > ("planes", 1);
    puby = nh.advertise < pcl::PCLPointCloud2 > ("cylinder", 1);

    pubm = nh.advertise < visualization_msgs::Marker > ("detected_cylinder", 1);

    cilinder_arr_pub = nh.advertise < visualization_msgs::MarkerArray > ("detected_cylinders", 0);

    // TODO: add custom message for informing
    brain_publisher = nh.advertise < geometry_msgs::Pose > ("cylinder_greet_instructions", 1);

    greet_publisher = nh.advertise<combined::CylinderGreetInstructions>("unique_cylinder_greet", 1000);

    // Spin
    ros::spin();
}
