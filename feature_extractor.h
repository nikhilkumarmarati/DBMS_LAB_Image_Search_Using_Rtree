#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>
#include <iostream>

class FeatureExtractor {
private:
    torch::jit::script::Module model;
    int feature_dim;
    
public:
    FeatureExtractor(const std::string& model_path) : feature_dim(25) {
        try {
            // Load the DINOv2 model
            model = torch::jit::load(model_path);
            model.eval();
            std::cout << "DINOv2 model loaded successfully" << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Error loading the model: " << e.what() << std::endl;
            throw;
        }
    }
    
    std::vector<double> extract(const cv::Mat& image) {
        // Resize and preprocess the image
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(224, 224));
        
        // Convert to RGB if needed
        cv::Mat rgb_image;
        if (resized.channels() == 1) {
            cv::cvtColor(resized, rgb_image, cv::COLOR_GRAY2RGB);
        } else if (resized.channels() == 3) {
            cv::cvtColor(resized, rgb_image, cv::COLOR_BGR2RGB);
        } else {
            rgb_image = resized;
        }
        
        // Normalize the image
        cv::Mat float_image;
        rgb_image.convertTo(float_image, CV_32F, 1.0/255.0);
        
        // // Normalize with ImageNet mean and std
        // cv::Scalar mean(0.485, 0.456, 0.406);
        // cv::Scalar std(0.229, 0.224, 0.225);
        // Normalize with STL10 mean and std
        cv::Scalar mean(0.4467, 0.4398, 0.4066);
        cv::Scalar std(0.2603, 0.2566, 0.2713);

        cv::Mat channels[3];
        cv::split(float_image, channels);
        
        for (int i = 0; i < 3; ++i) {
            channels[i] = (channels[i] - mean[i]) / std[i];
        }
        
        cv::merge(channels, 3, float_image);
        
        // Convert to tensor
        auto tensor_image = torch::from_blob(float_image.data, 
                                           {1, float_image.rows, float_image.cols, 3}, 
                                           torch::kFloat32);
        tensor_image = tensor_image.permute({0, 3, 1, 2});
        
        // Extract features
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(tensor_image);
        
        torch::NoGradGuard no_grad;
        auto output = model.forward(inputs).toTensor();
        
        // Convert to vector
        std::vector<double> features(feature_dim);
        auto accessor = output.accessor<float, 2>();
        
        for (int i = 0; i < feature_dim; ++i) {
            features[i] = accessor[0][i];
        }
        
        // Normalize the feature vector
        double norm = 0.0;
        for (int i = 0; i < feature_dim; ++i) {
            norm += features[i] * features[i];
        }
        norm = std::sqrt(norm);
        
        for (int i = 0; i < feature_dim; ++i) {
            features[i] /= norm;
        }
        
        return features;
    }
    
    int getFeatureDim() const {
        return feature_dim;
    }
};

#endif // FEATURE_EXTRACTOR_H
