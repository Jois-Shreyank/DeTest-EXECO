// main.cpp
//
// This program performs voxel-wise Gaussian Process (GP) smoothing on a PLY file.
// It reads vertices from an input PLY file, divides them into voxels (cells),
// applies GP regression to each cell to smooth the surface, and writes the
// resulting smoothed mesh to a new PLY file.
//
// Dependencies:
//   - Eigen: A C++ template library for linear algebra: matrices, vectors,
//            numerical solvers, and related algorithms.
//
// Compilation:
//   g++ -I /path/to/eigen/ main.cpp -o gp_meshing -std=c++14 -O3
//
// Usage:
//   ./gp_meshing <input.ply> <output.ply> [grid_size] [min_points_per_cell]
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

// Structure to hold a 3D point/vertex
struct Point {
    double x, y, z;
};

// Structure to represent a triangular face
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

    // Read header
    while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        std::string keyword;
        ss >> keyword;

        if (keyword == "element") {
            std::string type;
            int count;
            ss >> type >> count;
            if (type == "vertex") {
                vertex_count = count;
            } else if (type == "face") {
                face_count = count;
            }
        } else if (keyword == "end_header") {
            header_ended = true;
            break;
        }
    }

    if (!header_ended) {
        std::cerr << "Error: Invalid PLY header." << std::endl;
        return false;
    }

    // Read vertices
    vertices.reserve(vertex_count);
    for (int i = 0; i < vertex_count; ++i) {
        Point p;
        if (!(ifs >> p.x >> p.y >> p.z)) {
            std::cerr << "Error reading vertex " << i << std::endl;
            return false;
        }
        vertices.push_back(p);
    }

    // Read faces
    faces.reserve(face_count);
    for (int i = 0; i < face_count; ++i) {
        int num_vertices;
        Face f;
        if (!(ifs >> num_vertices >> f.v1 >> f.v2 >> f.v3)) {
            std::cerr << "Error reading face " << i << std::endl;
            return false;
        }
        if (num_vertices != 3) {
            std::cerr << "Warning: Non-triangular face detected. This program only supports triangular meshes." << std::endl;
        }
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

    // Write header
    ofs << "ply" << std::endl;
    ofs << "format ascii 1.0" << std::endl;
    ofs << "element vertex " << vertices.size() << std::endl;
    ofs << "property float x" << std::endl;
    ofs << "property float y" << std::endl;
    ofs << "property float z" << std::endl;
    ofs << "element face " << faces.size() << std::endl;
    ofs << "property list uchar int vertex_indices" << std::endl;
    ofs << "end_header" << std::endl;

    // Write vertices
    for (const auto& v : vertices) {
        ofs << v.x << " " << v.y << " " << v.z << std::endl;
    }

    // Write faces
    for (const auto& f : faces) {
        ofs << "3 " << f.v1 << " " << f.v2 << " " << f.v3 << std::endl;
    }
}

// Gaussian Process regression for a single cell
std::vector<Point> gaussianProcess(const Cell& cell, double grid_size) {
    int n_points = cell.points.size();
    if (n_points == 0) {
        return {};
    }

    // Determine the dominant direction via PCA
    Eigen::MatrixXd data(n_points, 3);
    for (int i = 0; i < n_points; ++i) {
        data(i, 0) = cell.points[i].x;
        data(i, 1) = cell.points[i].y;
        data(i, 2) = cell.points[i].z;
    }

    Eigen::Vector3d mean = data.colwise().mean();
    Eigen::MatrixXd centered = data.rowwise() - mean.transpose();
    Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(n_points - 1);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov);
    Eigen::Vector3d normal = eig.eigenvectors().col(0);

    int pred_dir = 0;
    if (std::abs(normal.y()) > std::abs(normal.x())) pred_dir = 1;
    if (std::abs(normal.z()) > std::abs(normal(pred_dir))) pred_dir = 2;

    int loc_dir1 = (pred_dir + 1) % 3;
    int loc_dir2 = (pred_dir + 2) % 3;

    Eigen::MatrixXd train_X(n_points, 2);
    Eigen::VectorXd train_Y(n_points);

    for (int i = 0; i < n_points; ++i) {
        train_X(i, 0) = data(i, loc_dir1);
        train_X(i, 1) = data(i, loc_dir2);
        train_Y(i) = data(i, pred_dir);
    }

    // GP parameters
    double kernel_length = 1.2;
    double variance_sensor = 0.1;

    // Create test points on a grid
    int num_test_side = 10;
    std::vector<Point> test_points;
    Point cell_min = cell.points[0];
    Point cell_max = cell.points[0];
    for(const auto& p : cell.points){
        if(p.x < cell_min.x) cell_min.x = p.x;
        if(p.y < cell_min.y) cell_min.y = p.y;
        if(p.z < cell_min.z) cell_min.z = p.z;
        if(p.x > cell_max.x) cell_max.x = p.x;
        if(p.y > cell_max.y) cell_max.y = p.y;
        if(p.z > cell_max.z) cell_max.z = p.z;
    }


    for (int i = 0; i < num_test_side; ++i) {
        for (int j = 0; j < num_test_side; ++j) {
            Point p;
            double u = (double)i / (num_test_side - 1);
            double v = (double)j / (num_test_side - 1);
            
            double coords[3] = {0,0,0};
            coords[loc_dir1] = cell_min.x + u * (cell_max.x - cell_min.x);
            coords[loc_dir2] = cell_min.y + v * (cell_max.y - cell_min.y);

            p.x = coords[0];
            p.y = coords[1];
            p.z = coords[2];

            test_points.push_back(p);
        }
    }

    Eigen::MatrixXd test_X(test_points.size(), 2);
    for(size_t i = 0; i < test_points.size(); ++i){
        double coords[3] = {test_points[i].x, test_points[i].y, test_points[i].z};
        test_X(i,0) = coords[loc_dir1];
        test_X(i,1) = coords[loc_dir2];
    }


    // GP regression
    Eigen::MatrixXd K(n_points, n_points);
    for (int i = 0; i < n_points; ++i) {
        for (int j = 0; j < n_points; ++j) {
            double dist = (train_X.row(i) - train_X.row(j)).norm();
            K(i, j) = exp(-kernel_length * dist);
        }
    }
    K += variance_sensor * variance_sensor * Eigen::MatrixXd::Identity(n_points, n_points);

    Eigen::MatrixXd K_star(test_X.rows(), n_points);
    for (int i = 0; i < test_X.rows(); ++i) {
        for (int j = 0; j < n_points; ++j) {
            double dist = (test_X.row(i) - train_X.row(j)).norm();
            K_star(i, j) = exp(-kernel_length * dist);
        }
    }

    Eigen::VectorXd pred_Y = K_star * K.llt().solve(train_Y);

    std::vector<Point> smoothed_points;
    for (int i = 0; i < pred_Y.size(); ++i) {
        Point p = test_points[i];
        double coords[3] = {p.x, p.y, p.z};
        coords[pred_dir] = pred_Y(i);
        smoothed_points.push_back({coords[0], coords[1], coords[2]});
    }

    return smoothed_points;
}


int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.ply> <output.ply> [grid_size] [min_points_per_cell]" << std::endl;
        return 1;
    }

    std::string input_filename = argv[1];
    std::string output_filename = argv[2];
    double grid_size = (argc > 3) ? std::stod(argv[3]) : 1.0;
    int min_points_per_cell = (argc > 4) ? std::stoi(argv[4]) : 10;

    // Read the input PLY file
    std::vector<Point> vertices;
    std::vector<Face> faces;
    if (!readPly(input_filename, vertices, faces)) {
        return 1;
    }
    std::cout << "Read " << vertices.size() << " vertices and " << faces.size() << " faces." << std::endl;

    // Voxelize the point cloud
    std::map<long long, Cell> grid;
    for (const auto& v : vertices) {
        long long ix = static_cast<long long>(floor(v.x / grid_size));
        long long iy = static_cast<long long>(floor(v.y / grid_size));
        long long iz = static_cast<long long>(floor(v.z / grid_size));
        long long key = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791);
        grid[key].points.push_back(v);
    }
    std::cout << "Voxelized into " << grid.size() << " cells." << std::endl;

    // Apply GP to each cell and collect smoothed points and faces
    std::vector<Point> smoothed_vertices;
    std::vector<Face> new_faces;
    for (auto const& [key, cell] : grid) {
        if (cell.points.size() >= min_points_per_cell) {
            std::vector<Point> cell_smoothed_points = gaussianProcess(cell, grid_size);
            
            int base_index = smoothed_vertices.size();
            smoothed_vertices.insert(smoothed_vertices.end(), cell_smoothed_points.begin(), cell_smoothed_points.end());

            // Generate faces for the cell's grid
            int num_test_side = 10; // Must match the value in gaussianProcess function
            for (int i = 0; i < num_test_side - 1; ++i) {
                for (int j = 0; j < num_test_side - 1; ++j) {
                    int v1 = base_index + i * num_test_side + j;
                    int v2 = base_index + i * num_test_side + j + 1;
                    int v3 = base_index + (i + 1) * num_test_side + j;
                    int v4 = base_index + (i + 1) * num_test_side + j + 1;

                    new_faces.push_back({v1, v3, v2});
                    new_faces.push_back({v2, v3, v4});
                }
            }
        } else {
            // If not enough points, just keep the original ones (no faces generated for these)
            smoothed_vertices.insert(smoothed_vertices.end(), cell.points.begin(), cell.points.end());
        }
    }
    std::cout << "Generated " << smoothed_vertices.size() << " smoothed vertices and " << new_faces.size() << " new faces." << std::endl;

    // Write the smoothed vertices and new faces to a PLY file
    writePly(output_filename, smoothed_vertices, new_faces);
    std::cout << "Smoothed mesh saved to " << output_filename << std::endl;

    return 0;
}
