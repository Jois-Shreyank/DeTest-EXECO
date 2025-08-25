#include <iostream>
#include <string>
#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/point_types.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>
#include "yaml-cpp/yaml.h" // Include YAML-CPP

// Define shorter type names for convenience
using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;
using NormalT = pcl::Normal;
using NormalCloudT = pcl::PointCloud<NormalT>;
using FPFHDescriptorT = pcl::FPFHSignature33;
using FPFHCloudT = pcl::PointCloud<FPFHDescriptorT>;
using SHOTDescriptorT = pcl::SHOT352;
using SHOTCloudT = pcl::PointCloud<SHOTDescriptorT>;

// --- Function Prototypes ---
NormalCloudT::Ptr compute_normals(const PointCloudT::Ptr& cloud);
PointCloudT::Ptr detect_iss_keypoints(const PointCloudT::Ptr& cloud, const YAML::Node& params);
PointCloudT::Ptr detect_harris_keypoints(const PointCloudT::Ptr& cloud, const NormalCloudT::Ptr& normals, const YAML::Node& params);
FPFHCloudT::Ptr compute_fpfh_descriptors(const PointCloudT::Ptr& cloud, const PointCloudT::Ptr& keypoints, const NormalCloudT::Ptr& normals, const YAML::Node& params);
SHOTCloudT::Ptr compute_shot_descriptors(const PointCloudT::Ptr& cloud, const PointCloudT::Ptr& keypoints, const NormalCloudT::Ptr& normals, const YAML::Node& params);

// --- Main Function ---
int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " mesh1.ply mesh2.ply <FPFH|SHOT> <ISS|HARRIS> config.yaml\n" << std::endl;
        return -1;
    }

    std::string mesh1_path = argv[1];
    std::string mesh2_path = argv[2];
    std::string descriptor_type = argv[3];
    std::string keypoint_type = argv[4];
    std::string config_path = argv[5];

    // --- 0. Load Configuration ---
    YAML::Node config;
    try {
        config = YAML::LoadFile(config_path);
    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading config file: " << e.what() << std::endl;
        return -1;
    }

    // --- 1. Load Files ---
    std::cout << "Loading meshes..." << std::endl;
    PointCloudT::Ptr cloud1(new PointCloudT);
    PointCloudT::Ptr cloud2(new PointCloudT);
    auto load_file = [](const std::string& path, PointCloudT::Ptr cloud) {
        if (path.rfind(".ply") != std::string::npos) return pcl::io::loadPLYFile<PointT>(path, *cloud);
        if (path.rfind(".obj") != std::string::npos) return pcl::io::loadOBJFile<PointT>(path, *cloud);
        std::cerr << "Unsupported file type: " << path << std::endl;
        return -1;
    };
    if (load_file(mesh1_path, cloud1) == -1 || load_file(mesh2_path, cloud2) == -1) {
        std::cerr << "Couldn't read one of the mesh files." << std::endl;
        return -1;
    }

    // --- 2. Compute Normals ---
    std::cout << "Computing normals..." << std::endl;
    NormalCloudT::Ptr normals1 = compute_normals(cloud1);
    NormalCloudT::Ptr normals2 = compute_normals(cloud2);

    // --- 3. Detect Keypoints ---
    std::cout << "Detecting keypoints using " << keypoint_type << "..." << std::endl;
    PointCloudT::Ptr keypoints1, keypoints2;
    const auto& kp_params = config["keypoint_detectors"];
    if (keypoint_type == "ISS") {
        keypoints1 = detect_iss_keypoints(cloud1, kp_params["ISS"]);
        keypoints2 = detect_iss_keypoints(cloud2, kp_params["ISS"]);
    } else if (keypoint_type == "HARRIS") {
        keypoints1 = detect_harris_keypoints(cloud1, normals1, kp_params["HARRIS"]);
        keypoints2 = detect_harris_keypoints(cloud2, normals2, kp_params["HARRIS"]);
    } else {
        std::cerr << "Unknown keypoint type: " << keypoint_type << std::endl;
        return -1;
    }
    std::cout << "Found " << keypoints1->size() << " keypoints in mesh 1 and " << keypoints2->size() << " in mesh 2." << std::endl;

    // --- 4. Compute Descriptors ---
    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences);
    const auto& desc_params = config["descriptors"];
    if (descriptor_type == "FPFH") {
        std::cout << "Computing FPFH descriptors..." << std::endl;
        FPFHCloudT::Ptr descriptors1 = compute_fpfh_descriptors(cloud1, keypoints1, normals1, desc_params["FPFH"]);
        FPFHCloudT::Ptr descriptors2 = compute_fpfh_descriptors(cloud2, keypoints2, normals2, desc_params["FPFH"]);
        pcl::registration::CorrespondenceEstimation<FPFHDescriptorT, FPFHDescriptorT> est;
        est.setInputSource(descriptors1);
        est.setInputTarget(descriptors2);
        est.determineReciprocalCorrespondences(*correspondences);
    } else if (descriptor_type == "SHOT") {
        std::cout << "Computing SHOT descriptors..." << std::endl;
        SHOTCloudT::Ptr descriptors1 = compute_shot_descriptors(cloud1, keypoints1, normals1, desc_params["SHOT"]);
        SHOTCloudT::Ptr descriptors2 = compute_shot_descriptors(cloud2, keypoints2, normals2, desc_params["SHOT"]);
        pcl::registration::CorrespondenceEstimation<SHOTDescriptorT, SHOTDescriptorT> est;
        est.setInputSource(descriptors1);
        est.setInputTarget(descriptors2);
        est.determineReciprocalCorrespondences(*correspondences);
    } else {
        std::cerr << "Unknown descriptor type: " << descriptor_type << std::endl;
        return -1;
    }
    std::cout << "Found " << correspondences->size() << " correspondences." << std::endl;

    // --- 5. Visualize the Results ---
    std::cout << "Visualizing results. Press 'q' to exit." << std::endl;
    pcl::visualization::PCLVisualizer viewer("Mesh Correspondences");
    viewer.setBackgroundColor(0.1, 0.1, 0.1);
    viewer.addCoordinateSystem(1.0);
    viewer.addPointCloud(cloud1, "cloud1");

    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform(1, 3) = 100.5; // Shift 100.5 meters on the y-axis
    PointCloudT::Ptr cloud2_transformed(new PointCloudT);
    pcl::transformPointCloud(*cloud2, *cloud2_transformed, transform);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> blue_color(cloud2_transformed, 0, 150, 255);
    viewer.addPointCloud(cloud2_transformed, blue_color, "cloud2");

    for (const auto& match : *correspondences) {
        const PointT& p1 = keypoints1->points[match.index_query];
        PointT p2_original = keypoints2->points[match.index_match];
        PointT p2_transformed;
        p2_transformed.x = p2_original.x;
        p2_transformed.y = p2_original.y + 100.5;
        p2_transformed.z = p2_original.z;

        viewer.addLine<PointT, PointT>(p1, p2_transformed, 0, 255, 0, "line_" + std::to_string(match.index_query));
    }
    viewer.spin();
    return 0;
}

// --- Function Implementations ---

NormalCloudT::Ptr compute_normals(const PointCloudT::Ptr& cloud) {
    NormalCloudT::Ptr normals(new NormalCloudT);
    pcl::NormalEstimationOMP<PointT, NormalT> norm_est;
    norm_est.setNumberOfThreads(8);
    norm_est.setKSearch(100);
    norm_est.setInputCloud(cloud);
    norm_est.compute(*normals);
    return normals;
}

PointCloudT::Ptr detect_iss_keypoints(const PointCloudT::Ptr& cloud, const YAML::Node& params) {
    PointCloudT::Ptr keypoints(new PointCloudT);
    pcl::ISSKeypoint3D<PointT, PointT> iss_detector;
    iss_detector.setSalientRadius(params["salient_radius"].as<double>());
    iss_detector.setNonMaxRadius(params["non_max_radius"].as<double>());
    iss_detector.setMinNeighbors(params["min_neighbors"].as<int>());
    iss_detector.setThreshold21(params["threshold21"].as<double>());
    iss_detector.setThreshold32(params["threshold32"].as<double>());
    iss_detector.setInputCloud(cloud);
    iss_detector.compute(*keypoints);
    return keypoints;
}

PointCloudT::Ptr detect_harris_keypoints(const PointCloudT::Ptr& cloud, const NormalCloudT::Ptr& normals, const YAML::Node& params) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints_with_intensity(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI> harris_detector;
    harris_detector.setNonMaxSupression(true);
    harris_detector.setRadius(params["radius"].as<double>());
    harris_detector.setMethod(pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI>::HARRIS);
    harris_detector.setInputCloud(cloud);
    harris_detector.setNormals(normals);
    harris_detector.compute(*keypoints_with_intensity);

    PointCloudT::Ptr keypoints(new PointCloudT);
    pcl::copyPointCloud(*keypoints_with_intensity, *keypoints);
    return keypoints;
}

FPFHCloudT::Ptr compute_fpfh_descriptors(const PointCloudT::Ptr& cloud, const PointCloudT::Ptr& keypoints, const NormalCloudT::Ptr& normals, const YAML::Node& params) {
    FPFHCloudT::Ptr descriptors(new FPFHCloudT);
    pcl::FPFHEstimationOMP<PointT, NormalT, FPFHDescriptorT> fpfh_est;
    fpfh_est.setNumberOfThreads(8);
    fpfh_est.setRadiusSearch(params["radius_search"].as<double>());
    fpfh_est.setInputCloud(keypoints);
    fpfh_est.setInputNormals(normals);
    fpfh_est.setSearchSurface(cloud);
    fpfh_est.compute(*descriptors);
    return descriptors;
}

SHOTCloudT::Ptr compute_shot_descriptors(const PointCloudT::Ptr& cloud, const PointCloudT::Ptr& keypoints, const NormalCloudT::Ptr& normals, const YAML::Node& params) {
    SHOTCloudT::Ptr descriptors(new SHOTCloudT);
    pcl::SHOTEstimationOMP<PointT, NormalT, SHOTDescriptorT> shot_est;
    shot_est.setNumberOfThreads(8);
    shot_est.setRadiusSearch(params["radius_search"].as<double>());
    shot_est.setInputCloud(keypoints);
    shot_est.setInputNormals(normals);
    shot_est.setSearchSurface(cloud);
    shot_est.compute(*descriptors);
    return descriptors;
}