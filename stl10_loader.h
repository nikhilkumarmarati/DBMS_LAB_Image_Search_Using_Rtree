#ifndef STL10_LOADER_H
#define STL10_LOADER_H

#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdexcept>

class STL10Loader {
private:
    std::string data_path;
    
public:
    STL10Loader(const std::string& path) : data_path(path) {}
    
    std::vector<cv::Mat> loadImages(const std::string& split, int max_images = -1) {
        std::string images_file;
        
        if (split == "train") {
            images_file = data_path + "/train_X.bin";
        } else if (split == "test") {
            images_file = data_path + "/test_X.bin";
        } else if (split == "unlabeled") {
            images_file = data_path + "/unlabeled_X.bin";
        } else {
            throw std::runtime_error("Invalid split name. Use 'train', 'test', or 'unlabeled'");
        }
        
        std::ifstream file(images_file, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Could not open file: " + images_file);
        }
        
        std::vector<cv::Mat> images;
        
        // STL-10 images are 96x96 RGB
        const int rows = 96;
        const int cols = 96;
        const int channels = 3;
        const int image_size = rows * cols * channels;
        
        std::vector<unsigned char> buffer(image_size);
        
        int count = 0;
        while (file.read(reinterpret_cast<char*>(buffer.data()), image_size)) {
            // STL-10 stores images in CHW format, need to convert to HWC
            cv::Mat image(rows, cols, CV_8UC3);
            
            for (int c = 0; c < channels; c++) {
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        image.at<cv::Vec3b>(i, j)[c] = buffer[c * rows * cols + i * cols + j];
                    }
                }
            }
            
            images.push_back(image);
            count++;
            
            if (max_images > 0 && count >= max_images) {
                break;
            }
        }
        
        std::cout << "Loaded " << images.size() << " images from " << split << " set" << std::endl;
        return images;
    }
    
    std::vector<int> loadLabels(const std::string& split) {
        if (split == "unlabeled") {
            throw std::runtime_error("Unlabeled set does not have labels");
        }
        
        std::string labels_file;
        
        if (split == "train") {
            labels_file = data_path + "/train_y.bin";
        } else if (split == "test") {
            labels_file = data_path + "/test_y.bin";
        } else {
            throw std::runtime_error("Invalid split name. Use 'train' or 'test'");
        }
        
        std::ifstream file(labels_file, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Could not open file: " + labels_file);
        }
        
        std::vector<int> labels;
        unsigned char label;
        
        while (file.read(reinterpret_cast<char*>(&label), sizeof(unsigned char))) {
            labels.push_back(static_cast<int>(label));
        }
        
        std::cout << "Loaded " << labels.size() << " labels from " << split << " set" << std::endl;
        return labels;
    }
};

#endif // STL10_LOADER_H
