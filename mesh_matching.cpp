#include <iostream>
#include <string>
#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/point_types.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>
// Define shorter type names for convenience
using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;
using NormalT = pcl::Normal;
using NormalCloudT = pcl::PointCloud<NormalT>;
using FPFHDescriptorT = pcl::FPFHSignature33;
using FPFHCloudT = pcl::PointCloud<FPFHDescriptorT>;
using SHOTDescriptorT = pcl::SHOT352;
using SHOTCloudT = pcl::PointCloud<SHOTDescriptorT>;


/**
 * @brief Computes surface normals for a given point cloud.
 * Normals describe the orientation of the surface at each point and are
 * required by most 3D feature descriptors.
 * @param cloud The input point cloud.
 * @return A point cloud containing the computed normals.
 */
NormalCloudT::Ptr compute_normals(const PointCloudT::Ptr& cloud) {
    NormalCloudT::Ptr normals(new NormalCloudT);
    pcl::NormalEstimationOMP<PointT, NormalT> norm_est;
    norm_est.setKSearch(100); // Use 100 nearest neighbors to estimate the normal
    norm_est.setInputCloud(cloud);
    norm_est.compute(*normals);
    return normals;
}

/**
 * @brief Detects keypoints (interesting points) on the mesh.
 * Instead of computing descriptors for every point, we focus on a smaller
 * set of salient keypoints to improve speed and distinctiveness.
 * @param cloud The input point cloud.
 * @return A point cloud containing the detected keypoints.
 */
PointCloudT::Ptr detect_iss_keypoints(const PointCloudT::Ptr& cloud) {
    PointCloudT::Ptr keypoints(new PointCloudT);
    pcl::ISSKeypoint3D<PointT, PointT> iss_detector;
    iss_detector.setSalientRadius(0.08);
    iss_detector.setNonMaxRadius(0.04);
    iss_detector.setMinNeighbors(10);
    iss_detector.setThreshold21(0.975);
    iss_detector.setThreshold32(0.975);
    iss_detector.setInputCloud(cloud);
    iss_detector.compute(*keypoints);
    return keypoints;
}

/**
 * @brief Detects keypoints using Harris3D. This detector needs normals.
 * Harris3D is good for finding corners.
 */
PointCloudT::Ptr detect_harris_keypoints(const PointCloudT::Ptr& cloud, const NormalCloudT::Ptr& normals) {
    // Harris detector outputs PointXYZI, so we need to create a cloud for that
    pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints_with_intensity(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI> harris_detector;
    harris_detector.setNonMaxSupression(true);
    harris_detector.setRadius(0.04);
    harris_detector.setMethod(pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI>::HARRIS);
    harris_detector.setInputCloud(cloud);
    harris_detector.setNormals(normals);
    harris_detector.compute(*keypoints_with_intensity);

    // Convert the PointXYZI cloud back to our standard PointXYZ cloud
    PointCloudT::Ptr keypoints(new PointCloudT);
    pcl::copyPointCloud(*keypoints_with_intensity, *keypoints);
    return keypoints;
}

// /**
//  * @brief Detects keypoints using SIFT3D. This detector needs normals.
//  * SIFT is good for finding scale-invariant features.
//  */
// PointCloudT::Ptr detect_sift_keypoints(const PointCloudT::Ptr& cloud, const NormalCloudT::Ptr& normals) {
//     // SIFT outputs PointWithScale, so we need a cloud for that
//     pcl::PointCloud<pcl::PointWithScale> keypoints_with_scale;
//     pcl::SIFTKeypoint<PointT, pcl::PointWithScale> sift_detector;
//     sift_detector.setInputCloud(cloud);
//     // These parameters are crucial and may need tuning for your specific data
//     sift_detector.setMinScale(0.01);
//     sift_detector.setNumberOfOctaves(4);
//     sift_detector.setNumberOfScalesPerOctave(5);
//     sift_detector.setMinimumContrast(0.005);
//     sift_detector.compute(keypoints_with_scale);

//     // Convert the PointWithScale cloud back to our standard PointXYZ cloud
//     PointCloudT::Ptr keypoints(new PointCloudT);
//     pcl::copyPointCloud(keypoints_with_scale, *keypoints);
//     return keypoints;
// }

/**
 * @brief Computes FPFH (Fast Point Feature Histogram) descriptors for keypoints.
 * FPFH is a robust local descriptor that captures the geometry around a keypoint.
 * @param cloud The full point cloud (surface).
 * @param keypoints The points for which to compute descriptors.
 * @param normals The surface normals.
 * @return A cloud of FPFH descriptors.
 */
FPFHCloudT::Ptr compute_fpfh_descriptors(const PointCloudT::Ptr& cloud, const PointCloudT::Ptr& keypoints, const NormalCloudT::Ptr& normals) {
    FPFHCloudT::Ptr descriptors(new FPFHCloudT);
    pcl::FPFHEstimationOMP<PointT, NormalT, FPFHDescriptorT> fpfh_est;
    fpfh_est.setRadiusSearch(0.08); // Use a search radius of 8cm
    fpfh_est.setInputCloud(keypoints);
    fpfh_est.setInputNormals(normals);
    fpfh_est.setSearchSurface(cloud);
    fpfh_est.compute(*descriptors);
    return descriptors;
}

/**
 * @brief Computes SHOT (Signatures of Histograms of OrienTations) descriptors.
 * SHOT is another powerful local descriptor, often more robust to noise.
 * @param cloud The full point cloud (surface).
 * @param keypoints The points for which to compute descriptors.
 * @param normals The surface normals.
 * @return A cloud of SHOT descriptors.
 */
SHOTCloudT::Ptr compute_shot_descriptors(const PointCloudT::Ptr& cloud, const PointCloudT::Ptr& keypoints, const NormalCloudT::Ptr& normals) {
    SHOTCloudT::Ptr descriptors(new SHOTCloudT);
    pcl::SHOTEstimationOMP<PointT, NormalT, SHOTDescriptorT> shot_est;
    shot_est.setRadiusSearch(0.08); // Use a search radius of 8cm
    shot_est.setInputCloud(keypoints);
    shot_est.setInputNormals(normals);
    shot_est.setSearchSurface(cloud);
    shot_est.compute(*descriptors);
    return descriptors;
}


int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " mesh1.ply mesh2.ply <FPFH|SHOT> <ISS|HARRIS|SIFT>\n" << std::endl;
        return -1;
    }

    std::string mesh1_path = argv[1];
    std::string mesh2_path = argv[2];
    std::string descriptor_type = argv[3];
    std::string keypoint_type = argv[4];

    // --- 1. Load Files ---
    std::cout << "Loading meshes..." << std::endl;
    PointCloudT::Ptr cloud1(new PointCloudT);
    PointCloudT::Ptr cloud2(new PointCloudT);

    // Lambda function to handle loading logic
    auto load_file = [](const std::string& path, PointCloudT::Ptr cloud) {
        if (path.rfind(".ply") != std::string::npos) {
            return pcl::io::loadPLYFile<PointT>(path, *cloud);
        } else if (path.rfind(".obj") != std::string::npos) {
            return pcl::io::loadOBJFile<PointT>(path, *cloud);
        } else {
            std::cerr << "Unsupported file type: " << path << std::endl;
            return -1;
        }
    };

    if (load_file(mesh1_path, cloud1) == -1 || load_file(mesh2_path, cloud2) == -1) {
        std::cerr << "Couldn't read one of the mesh files." << std::endl;
        return -1;
    }

    // --- ADD THIS DIAGNOSTIC BLOCK ---
    PointT min_pt1, max_pt1;
    pcl::getMinMax3D(*cloud1, min_pt1, max_pt1);
    std::cout << "--> Mesh 1 dimensions (WxHxD): "
              << max_pt1.x - min_pt1.x << " x "
              << max_pt1.y - min_pt1.y << " x "
              << max_pt1.z - min_pt1.z << std::endl;

    PointT min_pt2, max_pt2;
    pcl::getMinMax3D(*cloud2, min_pt2, max_pt2);
    std::cout << "--> Mesh 2 dimensions (WxHxD): "
              << max_pt2.x - min_pt2.x << " x "
              << max_pt2.y - min_pt2.y << " x "
              << max_pt2.z - min_pt2.z << std::endl;
    // --- END OF DIAGNOSTIC BLOCK ---

    
    // --- 2. Compute Normals ---
    std::cout << "Computing normals..." << std::endl;
    NormalCloudT::Ptr normals1 = compute_normals(cloud1);
    NormalCloudT::Ptr normals2 = compute_normals(cloud2);

    // --- 3. Detect Keypoints ---
    std::cout << "Detecting keypoints using " << keypoint_type << "..." << std::endl;
    PointCloudT::Ptr keypoints1(new PointCloudT);
    PointCloudT::Ptr keypoints2(new PointCloudT);

    // MODIFIED: Select keypoint detector based on argument
    if (keypoint_type == "ISS") {
        keypoints1 = detect_iss_keypoints(cloud1);
        keypoints2 = detect_iss_keypoints(cloud2);
    } else if (keypoint_type == "HARRIS") {
        keypoints1 = detect_harris_keypoints(cloud1, normals1);
        keypoints2 = detect_harris_keypoints(cloud2, normals2);
    // } else if (keypoint_type == "SIFT") {
    //     keypoints1 = detect_sift_keypoints(cloud1, normals1);
    //     keypoints2 = detect_sift_keypoints(cloud2, normals2);
    } else {
        std::cerr << "Unknown keypoint type: " << keypoint_type << ". Use ISS, HARRIS, or SIFT." << std::endl;
        return -1;
    }
    std::cout << "Found " << keypoints1->size() << " keypoints in mesh 1." << std::endl;
    std::cout << "Found " << keypoints2->size() << " keypoints in mesh 2." << std::endl;

    // --- 4. Compute Descriptors ---
    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences);

    if (descriptor_type == "FPFH") {
        std::cout << "Computing FPFH descriptors..." << std::endl;
        FPFHCloudT::Ptr descriptors1 = compute_fpfh_descriptors(cloud1, keypoints1, normals1);
        FPFHCloudT::Ptr descriptors2 = compute_fpfh_descriptors(cloud2, keypoints2, normals2);

        // --- 5. Find Correspondences ---
        std::cout << "Finding correspondences..." << std::endl;
        pcl::registration::CorrespondenceEstimation<FPFHDescriptorT, FPFHDescriptorT> est;
        est.setInputSource(descriptors1);
        est.setInputTarget(descriptors2);
        est.determineReciprocalCorrespondences(*correspondences);

    } else if (descriptor_type == "SHOT") {
        std::cout << "Computing SHOT descriptors..." << std::endl;
        SHOTCloudT::Ptr descriptors1 = compute_shot_descriptors(cloud1, keypoints1, normals1);
        SHOTCloudT::Ptr descriptors2 = compute_shot_descriptors(cloud2, keypoints2, normals2);

        // --- 5. Find Correspondences ---
        std::cout << "Finding correspondences..." << std::endl;
        pcl::registration::CorrespondenceEstimation<SHOTDescriptorT, SHOTDescriptorT> est;
        est.setInputSource(descriptors1);
        est.setInputTarget(descriptors2);
        est.determineReciprocalCorrespondences(*correspondences);

    } else {
        std::cerr << "Unknown descriptor type: " << descriptor_type << ". Use FPFH or SHOT." << std::endl;
        return -1;
    }

    std::cout << "Found " << correspondences->size() << " correspondences." << std::endl;

    // --- 6. Visualize the Results ---
    std::cout << "Visualizing results. Press 'q' to exit." << std::endl;
    pcl::visualization::PCLVisualizer viewer("Mesh Correspondences");
    viewer.setBackgroundColor(0.1, 0.1, 0.1);
    viewer.addCoordinateSystem(1.0);

    // Add the first mesh (white)
    viewer.addPointCloud(cloud1, "cloud1");

    // Add the second mesh, shifted to the right for better visualization
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform(1, 3) = 100.5; // Shift 1.5 meters on the x-axis
    pcl::PointCloud<PointT>::Ptr cloud2_transformed(new pcl::PointCloud<PointT>);
    pcl::transformPointCloud(*cloud2, *cloud2_transformed, transform);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> blue_color(cloud2_transformed, 0, 150, 255); // R, G, B
    viewer.addPointCloud(cloud2_transformed, blue_color, "cloud2");
    
    // Draw lines between the corresponding keypoints
    for (size_t i = 0; i < correspondences->size(); ++i) {
        const auto& match = (*correspondences)[i];
        const PointT& p1 = keypoints1->points[match.index_query];
        
        // We need to use the transformed point from cloud 2 for the line
        PointT p2_original = keypoints2->points[match.index_match];
        PointT p2_transformed;
        p2_transformed.x = p2_original.x;
        p2_transformed.y = p2_original.y + 100.5;
        p2_transformed.z = p2_original.z;

        std::string line_id = "line_" + std::to_string(i);
        viewer.addLine<PointT, PointT>(p1, p2_transformed, 0, 255, 0, line_id); // Green lines
    }

    viewer.spin();

    return 0;
}
