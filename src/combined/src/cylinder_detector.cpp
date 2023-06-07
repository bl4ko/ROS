#include <iostream>
#include <ros/console.h>
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
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>

// for laser projection
// laser_geometry::LaserProjection projector_;
// tf::TransformListener listener_

// for memory management check
#include <memory>
#include <unordered_map>

// for publishing greet instructions of detected cylinders to brain
ros::Publisher greet_publisher;

ros::Publisher pubx;
ros::Publisher puby;
ros::Publisher pubm;
ros::Publisher cilinder_arr_pub;

tf2_ros::Buffer tf2_buffer;

typedef pcl::PointXYZRGB PointT;

// number of votes for object position
std::vector<int> object_position_votes;
// current coordinates of detection
std::vector<geometry_msgs::Pose> potential_object_positions;
// true if position was already sent to brain
std::vector<bool> objects_already_sent;
// color of object at position i
std::vector<std::string> detected_object_colors;
// colors
std::vector<double> object_color_red;
std::vector<double> object_color_green;
std::vector<double> object_color_blue;

// how close the detections have to be to be considered the same object
double object_proximity_threshold = 1;
// how many detections before sending
// double object_num_detections_threshold = 2;
double object_num_detections_threshold = 5;

int marker_id = 0;

ros::Publisher brain_publisher;

visualization_msgs::MarkerArray cylinder_marker_arr;

double distance_euclidean(double x1, double y1, double x2, double y2)
{
    return sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2));
}

double distance_euclidean_pose(geometry_msgs::Pose pose1, geometry_msgs::Pose pose2)
{
    return distance_euclidean(pose1.position.x, pose1.position.y, pose2.position.x, pose2.position.y);
}

/*
Returns array of [r, g, b, a] that corresponds to string color
*/
std::vector<double> get_color_from_string(std::string color_str)
{
    std::unordered_map<std::string, std::vector<double>> color_map = {
        {"red", {1, 0, 0, 1}},
        {"green", {0, 1, 0, 1}},
        {"blue", {0, 0, 1, 1}},
        {"black", {0, 0, 0, 1}},
        {"yellow", {255, 165, 0, 1}}};

    auto it = color_map.find(color_str);
    if (it != color_map.end())
    {
        return it->second;
    }

    // Default: white color with alpha 0.9
    return {255, 255, 255, 0.9};
}

/*
Publishes a cylinder marker to rviz.
*/
void publish_new_confirmed_object(geometry_msgs::Pose pose, std::string color_str)
{

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

    marker.scale.x = 0.2;
    marker.scale.y = 0.2;
    marker.scale.z = 0.2;

    marker.color.r = marker_custom_color[0];
    marker.color.g = marker_custom_color[1];
    marker.color.b = marker_custom_color[2];
    marker.color.a = marker_custom_color[3];

    marker.lifetime = ros::Duration();

    cylinder_marker_arr.markers.push_back(marker);

    // pubm.publish(marker);
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
get_color(double red, double green, double blue)
{
    if (red > green && red > blue)
    {
        // Probably red
        // if(abs(red-green) < 20){
        if (abs(red - green) < 15)
        {
            // Probably yellow
            return "yellow";
        }

        return "red";
    }
    else if (green > red && green > blue)
    {
        // Probably green
        // if(abs(red-green) < 20){
        if (abs(red - green) < 15)
        {
            // Probably yellow
            return "yellow";
        }

        return "green";
    }
    else if (blue > red && blue > green)
    {
        // Probably blue
        return "blue";
    }

    return "unknown";
}

/*
    Manage detected objects. We search through all already detected cylinder groups, if the distance
    to any of those groups is small enough (< object_proximity_threshold) we add the new detection to
    that group.
    If object group has enough detections (>= object_num_detection_threshold) publish it.
*/
void new_potential_object(geometry_msgs::Pose pose, double red, double green, double blue)
{
    bool found_close_instance = false;
    // find closest object if available
    for (int i = 0; i < potential_object_positions.size(); i++)
    {
        std::cout << "Pose x: " << pose.position.x << " " << std::endl;
        std::cout << "Euclidean: " << distance_euclidean_pose(potential_object_positions[i], pose) << " " << std::endl;
        if (distance_euclidean_pose(potential_object_positions[i], pose) < object_proximity_threshold)
        {

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
            if ((object_position_votes[i] >= object_num_detections_threshold) && (objects_already_sent[i] == false))
            {
                // if ((object_position_votes[i] > object_num_detections_threshold)) {
                std::string color = get_color(object_color_red[i], object_color_green[i], object_color_blue[i]);

                if (color == "unknown")
                {
                    std::cout << "Unknown color" << std::endl;
                    continue;
                }

                publish_new_confirmed_object(potential_object_positions[i], color);
                objects_already_sent[i] = true;

                std::cout << "New one published AAAAAAAAAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBBBBBBBBBBBBBCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC." << std::endl;
            }
        }
    }

    int n = object_position_votes.size();
    for (int iii = 0; iii < n; iii++)
    {
        if (iii == n - 1)
        {
            std::cout << object_position_votes[iii] << std::endl;
            break;
        }

        std::cout << object_position_votes[iii] << ", ";
    }

    // if no closest found just append it
    if (!found_close_instance)
    {
        potential_object_positions.push_back(pose);
        object_position_votes.push_back(1);
        objects_already_sent.push_back(false);
        detected_object_colors.push_back("Unknown");

        object_color_red.push_back(red);
        object_color_green.push_back(green);
        object_color_blue.push_back(blue);
    }
}

/*
Filter out points in a point cloud that are not in a y specified range.
*/
void do_pass_through_filter(const pcl::PointCloud<PointT>::Ptr &cloud)
{
    pcl::PassThrough<PointT> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("y");    // Filter along the "y" axis
    pass.setFilterLimits(-0.3, 0.2); // Acceptable range of y-values between (-0.3, 0.2)
    pass.filter(*cloud);
}

/*
Estimate the normal vectors for each point in a point cloud.
*/
void get_normals_for_cloud(const pcl::PointCloud<PointT>::Ptr &cloud,
                           const pcl::PointCloud<pcl::Normal>::Ptr &cloud_normals)
{
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    pcl::NormalEstimation<PointT, pcl::Normal> ne;

    // Estimate point normals
    ne.setSearchMethod(tree);
    ne.setInputCloud(cloud);
    ne.setKSearch(50);
    ne.compute(*cloud_normals);
}

/*
Downsample the point cloud using voxel grid filtering.
*/
void downsample_pointcloud(const pcl::PCLPointCloud2ConstPtr &cloud_blob,
                           const pcl::PointCloud<PointT>::Ptr &cloud)
{
    // Create the filtering object. Voxel grid is a 3D grid where each cell in the
    // grid contains one or moe points from the point cloud
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    pcl::PCLPointCloud2::Ptr cloud_filtered_blob(new pcl::PCLPointCloud2);
    sor.setInputCloud(cloud_blob);
    // Dimension of the voxels in the grid (the size of cells in the grid)
    sor.setLeafSize(0.01f, 0.01f, 0.01f);
    // Perform downsampling
    sor.filter(*cloud_filtered_blob);
    pcl::fromPCLPointCloud2(*cloud_filtered_blob, *cloud);
}

/*
    Remove the normals of the inliers of a detected plane from the normal cloud.
*/
void extract_normals(const pcl::PointCloud<pcl::Normal>::Ptr &cloud_normals,
                     const pcl::PointIndices::Ptr &inliers_plane)
{

    pcl::ExtractIndices<pcl::Normal> extract_normals;
    extract_normals.setNegative(true);
    extract_normals.setInputCloud(cloud_normals);
    extract_normals.setIndices(inliers_plane);
    extract_normals.filter(*cloud_normals);
}

/**
 * @brief Callback function for processing depth-based 3D point cloud data.
 * This function identifies potential objects of interest in a 3D scene based on
 * depth, color, size, and geometric properties.
 *
 * @param cloud_blob The raw point cloud data to be processed.
 */
void cloud_cb_basic_depth(const pcl::PCLPointCloud2ConstPtr &cloud_blob)
{

    // Time stamp for ROS operation
    ros::Time time_rec, time_test;
    time_rec = ros::Time::now();

    // Declare PCL objects for filtering, normal estimation, segmentation,
    pcl::PassThrough<PointT> pass;
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
    pcl::PCDWriter writer;
    pcl::ExtractIndices<PointT> extract;
    pcl::ExtractIndices<pcl::Normal> extract_normals;
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    Eigen::Vector4f centroid;

    // Initialize Point Clouds for various stages of processing
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<PointT>::Ptr cloud_filtered2(new pcl::PointCloud<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2(new pcl::PointCloud<pcl::Normal>);
    pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients), coefficients_cylinder(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices), inliers_cylinder(new pcl::PointIndices);
    pcl::PointCloud<PointT>::Ptr cloud_filtered_depth(new pcl::PointCloud<PointT>);

    // Downsample the point cloud data
    downsample_pointcloud(cloud_blob, cloud);
    ROS_INFO("PointCloud has: %lu data points.", cloud->points.size());
    // writer.write<PointT>("./debug/cloud_downsampled.pcd", *cloud, false);

    // Filter the point cloud data based on depth (z-axis)
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0, 1.9);
    pass.filter(*cloud_filtered_depth);
    writer.write<PointT>("./debug/cloud_filtered_depth.pcd", *cloud_filtered_depth, false);

    // Filter the point cloud based on height (y-axis)
    pass.setInputCloud(cloud_filtered_depth);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-0.3, 0.2);
    pass.filter(*cloud_filtered);
    ROS_INFO("PointCloud after filtering has: %lu data points.", cloud_filtered->points.size());
    // writer.write<PointT>("./debug/height_filtered_cloud.pcd", *cloud_filtered, false);

    // If filtered cloud has insufficient points, exit
    if (cloud_filtered->points.size() <= 10)
    {
        return;
    }

    // Estimate surface normals for remaining points in the filtered point cloud
    ne.setSearchMethod(tree);
    ne.setInputCloud(cloud_filtered);
    ne.setKSearch(50);
    ne.compute(*cloud_normals);

    // Perform planar segmentation to identify the dominant plane in the point cloud
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
    seg.setNormalDistanceWeight(0.1);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.03);
    seg.setInputCloud(cloud_filtered);
    seg.setInputNormals(cloud_normals);
    // Obtain the plane inliers and coefficients
    seg.segment(*inliers_plane, *coefficients_plane);
    std::cout << "Plane coefficients: " << *coefficients_plane << std::endl;

    // Extract points in the plane from the point cloud
    extract.setInputCloud(cloud_filtered);
    extract.setIndices(inliers_plane);
    extract.setNegative(false);

    // Write the planar inliers to disk
    pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>());
    extract.filter(*cloud_plane);
    std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size() << " data points." << std::endl;

    // Remove the points in the plane, and keep the rest for further processing
    extract.setNegative(true);
    extract.filter(*cloud_filtered2);
    extract_normals.setNegative(true);
    extract_normals.setInputCloud(cloud_normals);
    extract_normals.setIndices(inliers_plane);
    extract_normals.filter(*cloud_normals2);

    // if there are no more points left, exit
    if (cloud_filtered2->points.size() <= 10)
    {
        return;
    }

    // Perform cylinder segmentation to identify largest cylindrical component
    // from remaining points
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_CYLINDER);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setNormalDistanceWeight(0.1);
    seg.setMaxIterations(10000);
    seg.setDistanceThreshold(0.05);
    seg.setRadiusLimits(0.11, 0.18);
    seg.setInputCloud(cloud_filtered2);
    seg.setInputNormals(cloud_normals2);
    // Obtain the cylinder inliers and coefficients
    seg.segment(*inliers_cylinder, *coefficients_cylinder);

    // if no cylinder is found exit
    if (coefficients_cylinder->values.size() == 0)
    {
        std::cout << "No cylinder found mmm: " << std::endl;
        return;
    }

    std::cout << "Cylinder coefficients: " << *coefficients_cylinder << std::endl;

    // Extract points in the cylinder from the remaining point cloud
    extract.setInputCloud(cloud_filtered2);
    extract.setIndices(inliers_cylinder);

    // Check validity of cylindrical component
    if (coefficients_cylinder->values[6] < 0.1 || coefficients_cylinder->values[6] > 0.13)
    {
        return;
    }

    std::cout << "Inliers_num_cylinder: " << inliers_cylinder->indices.size() << std::endl;
    if (inliers_cylinder->indices.size() < 1200)
    {
        return;
    }

    // If cylinder is found, calculate its centroid and color, and handle it as a potential object of interest
    extract.setNegative(false);
    pcl::PointCloud<PointT>::Ptr cloud_cylinder(new pcl::PointCloud<PointT>());
    extract.filter(*cloud_cylinder);
    if (cloud_cylinder->points.empty())
        std::cout << "Can't find the cylindrical component." << std::endl;
    else
    {
        std::cout << "PointCloud representing the cylindrical component: " << cloud_cylinder->points.size() << " data points." << std::endl;

        pcl::compute3DCentroid(*cloud_cylinder, centroid);
        std::cout << "centroid of the cylindrical component: " << centroid[0] << " " << centroid[1] << " " << centroid[2] << " " << centroid[3] << std::endl;

        // Create a point in the "camera_rgb_optical_frame"
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

        try
        {
            time_test = ros::Time::now();

            std::cout << time_rec << std::endl;
            std::cout << time_test << std::endl;
            tss = tf2_buffer.lookupTransform("map", "camera_rgb_optical_frame", time_rec);
            // tf2_buffer.transform(point_camera, point_map, "map", ros::Duration(2));
        }
        catch (tf2::TransformException &ex)
        {
            ROS_WARN("Transform warning: %s\n", ex.what());
        }

        // std::cout << tss ;

        tf2::doTransform(point_camera, point_map, tss);

        std::cout << "point_camera: " << point_camera.point.x << " " << point_camera.point.y << " " << point_camera.point.z << std::endl;

        std::cout << "point_map: " << point_map.point.x << " " << point_map.point.y << " " << point_map.point.z << std::endl;

        // too high point for cylinder
        if (point_map.point.z > 0.75)
        {
            std::cout << "Too high!" << std::endl;
            return;
        }

        // if on the floor
        if (point_map.point.z < 0.05)
        {
            std::cout << "Too low!" << std::endl;
            return;
        }

        double red = 0;
        double green = 0;
        double blue = 0;

        for (int iii = 0; iii < cloud_cylinder->points.size(); iii++)
        {
            red = red + cloud_cylinder->points[iii].r;
            green = green + cloud_cylinder->points[iii].g;
            blue = blue + cloud_cylinder->points[iii].b;
        }

        red = red / cloud_cylinder->points.size();
        green = green / cloud_cylinder->points.size();
        blue = blue / cloud_cylinder->points.size();

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
        if (pose_new.position.x != pose_new.position.x)
            return;

        if (pose_new.position.y != pose_new.position.y)
            return;

        if (pose_new.position.z != pose_new.position.z)
            return;

        // declare new cylinder detection and render it
        new_potential_object(pose_new, red, green, blue);
    }
}

int main(int argc, char **argv)
{

    // Initialize ROS
    ros::init(argc, argv, "cylinder_segment");
    ros::NodeHandle nh;

    // For transforming between coordinate frames
    tf2_ros::TransformListener tf2_listener(tf2_buffer);
    ros::Subscriber sub = nh.subscribe("/camera/depth/points", 1, cloud_cb_basic_depth);
    puby = nh.advertise<pcl::PCLPointCloud2>("cylinder", 1);
    pubm = nh.advertise<visualization_msgs::Marker>("detected_cylinder", 1);
    cilinder_arr_pub = nh.advertise<visualization_msgs::MarkerArray>("detected_cylinders", 0);
    brain_publisher = nh.advertise<geometry_msgs::Pose>("cylinder_greet_instructions", 1);
    greet_publisher = nh.advertise<combined::CylinderGreetInstructions>("unique_cylinder_greet", 1000);
    ros::spin();
}
