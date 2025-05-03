#include "rtree.h"
#include "feature_extractor.h"
#include "stl10_loader.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>
#include <cmath>

// Function to save feature vectors to a file
void saveFeatureVectors(const std::string& filename, const std::unordered_map<int, std::vector<double>>& features) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }
    
    size_t num_features = features.size();
    out.write(reinterpret_cast<const char*>(&num_features), sizeof(size_t));
    
    for (const auto& [id, feature] : features) {
        out.write(reinterpret_cast<const char*>(&id), sizeof(int));
        size_t feature_size = feature.size();
        out.write(reinterpret_cast<const char*>(&feature_size), sizeof(size_t));
        out.write(reinterpret_cast<const char*>(feature.data()), feature_size * sizeof(double));
    }
    
    out.close();
    std::cout << "Saved " << num_features << " feature vectors to " << filename << std::endl;
}

// Function to load feature vectors from a file
std::unordered_map<int, std::vector<double>> loadFeatureVectors(const std::string& filename) {
    std::unordered_map<int, std::vector<double>> features;
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Error: Could not open file " << filename << " for reading" << std::endl;
        return features;
    }
    
    size_t num_features;
    in.read(reinterpret_cast<char*>(&num_features), sizeof(size_t));
    
    for (size_t i = 0; i < num_features; i++) {
        int id;
        in.read(reinterpret_cast<char*>(&id), sizeof(int));
        
        size_t feature_size;
        in.read(reinterpret_cast<char*>(&feature_size), sizeof(size_t));
        
        std::vector<double> feature(feature_size);
        in.read(reinterpret_cast<char*>(feature.data()), feature_size * sizeof(double));
        
        features[id] = feature;
    }
    
    in.close();
    std::cout << "Loaded " << features.size() << " feature vectors from " << filename << std::endl;
    return features;
}

// Function to save the R-tree to a file
void saveRTree(const std::string& filename, RTree& rtree) {
    try {
        rtree.save(filename);
        std::cout << "Saved R-tree to " << filename << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error saving R-tree: " << e.what() << std::endl;
    }
}

// Function to load the R-tree from a file
RTree loadRTree(const std::string& filename, int max_entries = 10) {
    RTree rtree(max_entries);
    try {
        rtree.load(filename);
        std::cout << "Loaded R-tree from " << filename << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading R-tree: " << e.what() << std::endl;
    }
    return rtree;
}

// Function to extract features from images
std::unordered_map<int, std::vector<double>> extractFeatures(const std::vector<cv::Mat>& images, FeatureExtractor& extractor) {
    std::unordered_map<int, std::vector<double>> features;
    
    for (size_t i = 0; i < images.size(); i++) {
        std::vector<double> feature = extractor.extract(images[i]);
        features[i] = feature;
        
        if ((i + 1) % 100 == 0 || i == images.size() - 1) {
            std::cout << "Extracted features for " << (i + 1) << " / " << images.size() << " images" << std::endl;
        }
    }
    
    return features;
}

// Function to build the R-tree from feature vectors
RTree buildRTree(const std::unordered_map<int, std::vector<double>>& features, int max_entries = 10) {
    RTree rtree(max_entries);
    
    size_t count = 0;
    for (const auto& [id, feature] : features) {
        rtree.insert(id, feature);
        count++;
        
        if (count % 100 == 0 || count == features.size()) {
            std::cout << "Inserted " << count << " / " << features.size() << " vectors into R-tree" << std::endl;
        }
    }
    
    return rtree;
}

// Function to evaluate the R-tree with k-NN queries
void evaluateRTree(RTree& rtree, const std::unordered_map<int, std::vector<double>>& features, int k = 10, int num_queries = 100) {
    if (features.size() < static_cast<size_t>(num_queries)) {
        std::cerr << "Error: Not enough feature vectors for evaluation" << std::endl;
        return;
    }
    
    // Select random query points
    std::vector<int> query_ids;
    std::vector<int> all_ids;
    
    for (const auto& [id, _] : features) {
        all_ids.push_back(id);
    }
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(all_ids.begin(), all_ids.end(), g);
    
    query_ids.insert(query_ids.end(), all_ids.begin(), all_ids.begin() + num_queries);
    
    // Perform k-NN queries
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int query_id : query_ids) {
        const auto& query_point = features.at(query_id);
        auto results = rtree.kNearestNeighbors(query_point, k);
        
        std::cout << "Query ID: " << query_id << ", Results: ";
        for (const auto& [id, dist] : results) {
            std::cout << "(" << id << ", " << dist << ") ";
        }
        std::cout << std::endl;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Performed " << num_queries << " k-NN queries with k=" << k << " in " << duration.count() << " ms" << std::endl;
    std::cout << "Average query time: " << (duration.count() / num_queries) << " ms" << std::endl;
}

int main(int argc, char** argv) {
    // Parse command line arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <mode> <data_path> [model_path] [rtree_file] [features_file]" << std::endl;
        std::cerr << "Modes: extract, build, load, query" << std::endl;
        return 1;
    }
    
    std::string mode = argv[1];
    std::string data_path = argv[2];
    std::string model_path = (argc > 3) ? argv[3] : "dinov2_pca_25d.pt";
    std::string rtree_file = (argc > 4) ? argv[4] : "rtree.bin";
    std::string features_file = (argc > 5) ? argv[5] : "features.bin";
    
    if (mode == "extract") {
        // Load STL-10 dataset
        STL10Loader loader(data_path);
        std::vector<cv::Mat> images = loader.loadImages("train");
        
        // Initialize feature extractor
        FeatureExtractor extractor(model_path);
        
        // Extract features
        auto features = extractFeatures(images, extractor);
        
        // Save features to file
        saveFeatureVectors(features_file, features);
    }
    else if (mode == "build") {
        // Load feature vectors
        auto features = loadFeatureVectors(features_file);
        
        // Build R-tree
        RTree rtree = buildRTree(features);
        
        // Save R-tree to file
        saveRTree(rtree_file, rtree);
    }
    else if (mode == "load") {
        // Load R-tree from file
        RTree rtree = loadRTree(rtree_file);
        
        // Load feature vectors
        auto features = loadFeatureVectors(features_file);
        
        // Evaluate R-tree
        evaluateRTree(rtree, features);
    }
    else if (mode == "query") {
        // Load R-tree from file
        RTree rtree = loadRTree(rtree_file);
        
        // Load feature vectors
        auto features = loadFeatureVectors(features_file);
        
        // Load STL-10 dataset for queries
        STL10Loader loader(data_path);
        std::vector<cv::Mat> query_images = loader.loadImages("test", 100);
        
        // Initialize feature extractor
        FeatureExtractor extractor(model_path);
        
        // Extract features for query images
        auto query_features = extractFeatures(query_images, extractor);

        //start the timer
        auto start_time = std::chrono::high_resolution_clock::now();
        
        int count = 1;
        // Perform k-NN queries
        for (const auto& [id, query_point] : query_features) {
            std::cout << "Query image " << count++ << ":" << std::endl;
            auto results = rtree.kNearestNeighbors(query_point, 5);
            
            std::cout << "Top 5 nearest neighbors:" << std::endl;
            for (const auto& [result_id, dist] : results) {
                std::cout << "Image ID: " << result_id << ", Distance: " << dist << std::endl;
            }
            std::cout << std::endl;
        }

        //stop the timer
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Total query time: " << duration.count() << " ms" << std::endl;
        std::cout << "Average query time: " << (duration.count() / (float)query_features.size()) << " ms" << std::endl;
    }
    else {
        std::cerr << "Invalid mode: " << mode << std::endl;
        return 1;
    }
    
    return 0;
}
