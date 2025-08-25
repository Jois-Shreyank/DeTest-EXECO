// main.cpp
//
// This program performs voxel-wise Gaussian Process (GP) smoothing on a PLY file
// and generates a mesh by creating faces within each processed voxel, similar to SLAMesh.
//
// Dependencies:
//   - Eigen: A C++ template library for linear algebra.
//
// Compilation:
//   g++ -I /path/to/eigen/ main.cpp -o gp_meshing -std=c++17 -O3
//
// Usage:
//   ./gp_meshing <input.ply> <output.ply> [voxel_size] [min_points_per_voxel]
//
// Example:
//   ./gp_meshing global_mesh_18-08-2025_Neighbourhood.ply output_smoothed.ply 1.0 10

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <map>
#include <Eigen/Dense>
#include <algorithm>
#include <limits>

// Basic data structures for 3D points and faces
struct Point {
    double x, y, z;
};

struct Face {
    int v1, v2, v3;
};

// Represents a cell in the voxel grid, containing the points within it
struct Cell {
    std::vector<Point> points;
};

// Function to read vertices and faces from a PLY file
bool readPly(const std::string& filename, std::vector<Point>& vertices, std::vector<Face>& faces) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    std::string line;
    int vertex_count = 0;
    int face_count = 0;
    bool header_ended = false;

    while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        std::string keyword;
        ss >> keyword;
        if (keyword == "element") {
            std::string type;
            int count;
            ss >> type >> count;
            if (type == "vertex") vertex_count = count;
            else if (type == "face") face_count = count;
        } else if (keyword == "end_header") {
            header_ended = true;
            break;
        }
    }

    if (!header_ended) {
        std::cerr << "Error: Invalid PLY header." << std::endl;
        return false;
    }

    vertices.reserve(vertex_count);
    for (int i = 0; i < vertex_count; ++i) {
        Point p;
        if (!(ifs >> p.x >> p.y >> p.z)) return false;
        vertices.push_back(p);
    }

    faces.reserve(face_count);
    for (int i = 0; i < face_count; ++i) {
        int num_vertices;
        Face f;
        if (!(ifs >> num_vertices >> f.v1 >> f.v2 >> f.v3)) return false;
        if (num_vertices != 3) continue;
        faces.push_back(f);
    }
    return true;
}

// Function to write vertices and faces to a PLY file
void writePly(const std::string& filename, const std::vector<Point>& vertices, const std::vector<Face>& faces) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }
    ofs << "ply" << std::endl;
    ofs << "format ascii 1.0" << std::endl;
    ofs << "element vertex " << vertices.size() << std::endl;
    ofs << "property float x" << std::endl;
    ofs << "property float y" << std::endl;
    ofs << "property float z" << std::endl;
    ofs << "element face " << faces.size() << std::endl;
    ofs << "property list uchar int vertex_indices" << std::endl;
    ofs << "end_header" << std::endl;
    for (const auto& v : vertices) {
        ofs << v.x << " " << v.y << " " << v.z << std::endl;
    }
    for (const auto& f : faces) {
        ofs << "3 " << f.v1 << " " << f.v2 << " " << f.v3 << std::endl;
    }
}

// Gaussian Process regression for a single cell, predicting along a specified direction
std::vector<Point> gaussianProcess(const Cell& cell, int pred_dir) {
    int n_points = cell.points.size();
    if (n_points == 0) return {};

    int loc_dir1 = (pred_dir + 1) % 3;
    int loc_dir2 = (pred_dir + 2) % 3;

    Eigen::MatrixXd train_X(n_points, 2);
    Eigen::VectorXd train_Y(n_points);

    for (int i = 0; i < n_points; ++i) {
        double coords[] = {cell.points[i].x, cell.points[i].y, cell.points[i].z};
        train_X(i, 0) = coords[loc_dir1];
        train_X(i, 1) = coords[loc_dir2];
        train_Y(i) = coords[pred_dir];
    }

    double kernel_length = 1.2;
    double variance_sensor = 0.1;

    int num_test_side = 10;
    Point cell_min = cell.points[0], cell_max = cell.points[0];
    for(const auto& p : cell.points){
        cell_min.x = std::min(cell_min.x, p.x); cell_min.y = std::min(cell_min.y, p.y); cell_min.z = std::min(cell_min.z, p.z);
        cell_max.x = std::max(cell_max.x, p.x); cell_max.y = std::max(cell_max.y, p.y); cell_max.z = std::max(cell_max.z, p.z);
    }
    double min_coords[] = {cell_min.x, cell_min.y, cell_min.z};
    double max_coords[] = {cell_max.x, cell_max.y, cell_max.z};

    Eigen::MatrixXd test_X(num_test_side * num_test_side, 2);
    for (int i = 0; i < num_test_side; ++i) {
        for (int j = 0; j < num_test_side; ++j) {
            double u = (double)i / (num_test_side - 1);
            double v = (double)j / (num_test_side - 1);
            test_X(i * num_test_side + j, 0) = min_coords[loc_dir1] + u * (max_coords[loc_dir1] - min_coords[loc_dir1]);
            test_X(i * num_test_side + j, 1) = min_coords[loc_dir2] + v * (max_coords[loc_dir2] - min_coords[loc_dir2]);
        }
    }
    
    Eigen::MatrixXd K(n_points, n_points);
    for (int i = 0; i < n_points; ++i) {
        for (int j = 0; j < n_points; ++j) {
            K(i, j) = exp(-kernel_length * (train_X.row(i) - train_X.row(j)).norm());
        }
    }
    K += variance_sensor * variance_sensor * Eigen::MatrixXd::Identity(n_points, n_points);
    Eigen::LLT<Eigen::MatrixXd> llt(K);

    Eigen::MatrixXd K_star(test_X.rows(), n_points);
    for (int i = 0; i < test_X.rows(); ++i) {
        for (int j = 0; j < n_points; ++j) {
            K_star(i, j) = exp(-kernel_length * (test_X.row(i) - train_X.row(j)).norm());
        }
    }

    Eigen::VectorXd pred_Y = K_star * llt.solve(train_Y);

    std::vector<Point> smoothed_points;
    for (int i = 0; i < pred_Y.size(); ++i) {
        Point p = {0,0,0};
        double coords[3] = {0,0,0};
        coords[loc_dir1] = test_X(i, 0);
        coords[loc_dir2] = test_X(i, 1);
        coords[pred_dir] = pred_Y(i);
        smoothed_points.push_back({coords[0], coords[1], coords[2]});
    }
    return smoothed_points;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.ply> <output.ply> [voxel_size] [min_points_per_voxel]" << std::endl;
        return 1;
    }

    std::string input_filename = argv[1];
    std::string output_filename = argv[2];
    double voxel_size = (argc > 3) ? std::stod(argv[3]) : 1.0;
    int min_points_per_voxel = (argc > 4) ? std::stoi(argv[4]) : 10;

    std::vector<Point> vertices;
    std::vector<Face> faces;
    if (!readPly(input_filename, vertices, faces)) return 1;
    std::cout << "Read " << vertices.size() << " vertices." << std::endl;

    std::map<long long, Cell> grid;
    for (const auto& v : vertices) {
        long long ix = static_cast<long long>(floor(v.x / voxel_size));
        long long iy = static_cast<long long>(floor(v.y / voxel_size));
        long long iz = static_cast<long long>(floor(v.z / voxel_size));
        long long key = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791);
        grid[key].points.push_back(v);
    }
    std::cout << "Voxelized into " << grid.size() << " cells." << std::endl;

    std::vector<Point> final_vertices;
    std::vector<Face> final_faces;

    for (auto const& [key, cell] : grid) {
        if (cell.points.size() >= min_points_per_voxel) {
            Eigen::MatrixXd data(cell.points.size(), 3);
            for (size_t i = 0; i < cell.points.size(); ++i) {
                data(i, 0) = cell.points[i].x; data(i, 1) = cell.points[i].y; data(i, 2) = cell.points[i].z;
            }
            Eigen::Vector3d mean = data.colwise().mean();
            Eigen::MatrixXd centered = data.rowwise() - mean.transpose();
            Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(cell.points.size() - 1);
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov);
            
            int pred_dir = 0;
            eig.eigenvectors().col(0).cwiseAbs().maxCoeff(&pred_dir);
            
            std::vector<Point> cell_smoothed_points = gaussianProcess(cell, pred_dir);
            
            int base_index = final_vertices.size();
            final_vertices.insert(final_vertices.end(), cell_smoothed_points.begin(), cell_smoothed_points.end());

            int num_test_side = 10;
            for (int i = 0; i < num_test_side - 1; ++i) {
                for (int j = 0; j < num_test_side - 1; ++j) {
                    int v1 = base_index + i * num_test_side + j;
                    int v2 = base_index + i * num_test_side + j + 1;
                    int v3 = base_index + (i + 1) * num_test_side + j;
                    int v4 = base_index + (i + 1) * num_test_side + j + 1;
                    final_faces.push_back({v1, v3, v2});
                    final_faces.push_back({v2, v3, v4});
                }
            }
        }
    }
    std::cout << "Generated " << final_vertices.size() << " vertices and " << final_faces.size() << " faces." << std::endl;

    writePly(output_filename, final_vertices, final_faces);
    std::cout << "Smoothed mesh saved to " << output_filename << std::endl;

    return 0;
}
