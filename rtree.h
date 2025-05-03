#ifndef RTREE_H
#define RTREE_H

#include <algorithm>
#include <vector>
#include <limits>
#include <cmath>
#include <memory>
#include <iostream>
#include <queue>
#include <functional>
#include <fstream>
#include <string>
#include <unordered_map>

// Define the dimensionality for our feature vectors
const int DIMENSIONS = 25; // Update if using 384-dimensional features

// Rectangle class to represent bounding boxes in high-dimensional space
class Rectangle {
public:
    std::vector<double> min;
    std::vector<double> max;
    
    Rectangle() : min(DIMENSIONS, std::numeric_limits<double>::max()), 
                  max(DIMENSIONS, std::numeric_limits<double>::lowest()) {}
    
    Rectangle(const std::vector<double>& point) : min(point), max(point) {}
    
    Rectangle(const std::vector<double>& min_point, const std::vector<double>& max_point) 
        : min(min_point), max(max_point) {}
    
    double area() const {
        double result = 1.0;
        for (int i = 0; i < DIMENSIONS; i++) {
            result *= (max[i] - min[i] + 1e-10); // Add small epsilon to avoid zero area
        }
        return result;
    }
    
    double margin() const {
        double result = 0.0;
        for (int i = 0; i < DIMENSIONS; i++) {
            result += (max[i] - min[i]);
        }
        return result;
    }
    
    bool contains(const Rectangle& other) const {
        for (int i = 0; i < DIMENSIONS; i++) {
            if (other.min[i] < min[i] || other.max[i] > max[i]) {
                return false;
            }
        }
        return true;
    }
    
    bool intersects(const Rectangle& other) const {
        for (int i = 0; i < DIMENSIONS; i++) {
            if (min[i] > other.max[i] || max[i] < other.min[i]) {
                return false;
            }
        }
        return true;
    }
    
    void extend(const Rectangle& other) {
        for (int i = 0; i < DIMENSIONS; i++) {
            min[i] = std::min(min[i], other.min[i]);
            max[i] = std::max(max[i], other.max[i]);
        }
    }
    
    double minDist(const std::vector<double>& point) const {
        double dist = 0.0;
        for (int i = 0; i < DIMENSIONS; i++) {
            if (point[i] < min[i]) {
                dist += (min[i] - point[i]) * (min[i] - point[i]);
            } else if (point[i] > max[i]) {
                dist += (point[i] - max[i]) * (point[i] - max[i]);
            }
        }
        return dist;
    }
    
    // Serialization methods
    void serialize(std::ofstream& out) const {
        for (int i = 0; i < DIMENSIONS; i++) {
            out.write(reinterpret_cast<const char*>(&min[i]), sizeof(double));
        }
        for (int i = 0; i < DIMENSIONS; i++) {
            out.write(reinterpret_cast<const char*>(&max[i]), sizeof(double));
        }
    }
    
    void deserialize(std::ifstream& in) {
        for (int i = 0; i < DIMENSIONS; i++) {
            in.read(reinterpret_cast<char*>(&min[i]), sizeof(double));
        }
        for (int i = 0; i < DIMENSIONS; i++) {
            in.read(reinterpret_cast<char*>(&max[i]), sizeof(double));
        }
    }
};

// Forward declaration
class Node;

// Entry class to store data in the R-tree
class Entry {
public:
    Rectangle mbr;
    int data_id;
    
    Entry() : data_id(-1) {}
    Entry(const Rectangle& mbr, int data_id) : mbr(mbr), data_id(data_id) {}
    
    // Serialization methods
    void serialize(std::ofstream& out) const {
        mbr.serialize(out);
        out.write(reinterpret_cast<const char*>(&data_id), sizeof(int));
    }
    
    void deserialize(std::ifstream& in) {
        mbr.deserialize(in);
        in.read(reinterpret_cast<char*>(&data_id), sizeof(int));
    }
};

// Node class for the R-tree
class Node {
public:
    bool is_leaf;
    Rectangle mbr;
    std::vector<std::shared_ptr<Node>> children;
    std::vector<Entry> entries;
    std::weak_ptr<Node> parent;
    
    Node(bool is_leaf) : is_leaf(is_leaf) {}
    
    void updateMBR() {
        if (is_leaf) {
            if (entries.empty()) {
                mbr = Rectangle(); // Reset to default
                return;
            }
            
            mbr = entries[0].mbr;
            for (size_t i = 1; i < entries.size(); i++) {
                mbr.extend(entries[i].mbr);
            }
        } else {
            if (children.empty()) {
                mbr = Rectangle(); // Reset to default
                return;
            }
            
            mbr = children[0]->mbr;
            for (size_t i = 1; i < children.size(); i++) {
                mbr.extend(children[i]->mbr);
            }
        }
    }
    
    // Serialization methods
    void serialize(std::ofstream& out) const {
        out.write(reinterpret_cast<const char*>(&is_leaf), sizeof(bool));
        mbr.serialize(out);
        
        if (is_leaf) {
            size_t entries_size = entries.size();
            out.write(reinterpret_cast<const char*>(&entries_size), sizeof(size_t));
            for (const auto& entry : entries) {
                entry.serialize(out);
            }
        } else {
            size_t children_size = children.size();
            out.write(reinterpret_cast<const char*>(&children_size), sizeof(size_t));
            for (const auto& child : children) {
                child->serialize(out);
            }
        }
    }
    
    void deserialize(std::ifstream& in, std::shared_ptr<Node> self) {
        in.read(reinterpret_cast<char*>(&is_leaf), sizeof(bool));
        mbr.deserialize(in);
        
        if (is_leaf) {
            size_t entries_size;
            in.read(reinterpret_cast<char*>(&entries_size), sizeof(size_t));
            entries.resize(entries_size);
            for (size_t i = 0; i < entries_size; i++) {
                entries[i].deserialize(in);
            }
        } else {
            size_t children_size;
            in.read(reinterpret_cast<char*>(&children_size), sizeof(size_t));
            children.resize(children_size);
            for (size_t i = 0; i < children_size; i++) {
                children[i] = std::make_shared<Node>(false);
                children[i]->parent = self;
                children[i]->deserialize(in, children[i]);
            }
        }
    }
};

// R-tree class
class RTree {
private:
    std::shared_ptr<Node> root;
    int max_entries;
    int min_entries;
    std::unordered_map<int, std::vector<double>> data_points;
    
    std::shared_ptr<Node> chooseLeaf(std::shared_ptr<Node> node, const Rectangle& mbr) {
        if (node->is_leaf) {
            return node;
        }
        
        double min_increase = std::numeric_limits<double>::max();
        std::shared_ptr<Node> best_child = nullptr;
        
        for (auto& child : node->children) {
            double area_before = child->mbr.area();
            Rectangle extended_mbr = child->mbr;
            extended_mbr.extend(mbr);
            double area_after = extended_mbr.area();
            double increase = area_after - area_before;
            
            if (best_child == nullptr || increase < min_increase || 
                (std::abs(increase - min_increase) < 1e-10 && child->mbr.area() < best_child->mbr.area())) {
                min_increase = increase;
                best_child = child;
            }
        }
        
        return chooseLeaf(best_child, mbr);
    }
    
    void adjustTree(std::shared_ptr<Node> node) {
        node->updateMBR();
        
        auto parent = node->parent.lock();
        if (parent) {
            adjustTree(parent);
        }
    }
    
    std::shared_ptr<Node> findParent(std::shared_ptr<Node> node, std::shared_ptr<Node> child) {
        if (node->is_leaf) return nullptr;
        
        for (auto& n : node->children) {
            if (n == child) return node;
            
            auto result = findParent(n, child);
            if (result) return result;
        }
        
        return nullptr;
    }
    
    void splitNode(std::shared_ptr<Node> node) {
        if (node->is_leaf) {
            splitLeafNode(node);
        } else {
            splitNonLeafNode(node);
        }
    }
    
    void splitLeafNode(std::shared_ptr<Node> node) {
        // Create a new node
        auto new_node = std::make_shared<Node>(true);
        
        // Find the seeds for splitting
        int seed1 = 0, seed2 = 0;
        findSeeds(node->entries, seed1, seed2);
        
        // Create two groups
        std::vector<Entry> group1, group2;
        group1.push_back(node->entries[seed1]);
        group2.push_back(node->entries[seed2]);
        
        // Create a list of entries to distribute
        std::vector<Entry> remaining;
        for (size_t i = 0; i < node->entries.size(); i++) {
            if (i != seed1 && i != seed2) {
                remaining.push_back(node->entries[i]);
            }
        }
        
        // Calculate initial MBRs
        Rectangle mbr1 = group1[0].mbr;
        Rectangle mbr2 = group2[0].mbr;
        
        // Distribute remaining entries
        while (!remaining.empty()) {
            // If one group has too few entries, assign all remaining to it
            if (group1.size() + remaining.size() <= min_entries) {
                group1.insert(group1.end(), remaining.begin(), remaining.end());
                break;
            }
            if (group2.size() + remaining.size() <= min_entries) {
                group2.insert(group2.end(), remaining.begin(), remaining.end());
                break;
            }
            
            // Find the entry that has the maximum difference in area increase
            int selected = 0;
            double max_diff = -std::numeric_limits<double>::max();
            int preferred_group = 0;
            
            for (size_t i = 0; i < remaining.size(); i++) {
                Rectangle extended1 = mbr1;
                extended1.extend(remaining[i].mbr);
                double increase1 = extended1.area() - mbr1.area();
                
                Rectangle extended2 = mbr2;
                extended2.extend(remaining[i].mbr);
                double increase2 = extended2.area() - mbr2.area();
                
                double diff = std::abs(increase1 - increase2);
                
                if (diff > max_diff) {
                    max_diff = diff;
                    selected = i;
                    preferred_group = (increase1 < increase2) ? 1 : 2;
                }
            }
            
            // Assign the selected entry to the preferred group
            if (preferred_group == 1) {
                group1.push_back(remaining[selected]);
                mbr1.extend(remaining[selected].mbr);
            } else {
                group2.push_back(remaining[selected]);
                mbr2.extend(remaining[selected].mbr);
            }
            
            // Remove the selected entry from the remaining list
            remaining.erase(remaining.begin() + selected);
        }
        
        // Update the nodes
        node->entries = group1;
        new_node->entries = group2;
        
        // Update MBRs
        node->updateMBR();
        new_node->updateMBR();
        
        // Handle the parent
        auto parent = node->parent.lock();
        if (!parent) {
            // This is the root, create a new root
            auto new_root = std::make_shared<Node>(false);
            new_root->children.push_back(node);
            new_root->children.push_back(new_node);
            node->parent = new_root;
            new_node->parent = new_root;
            new_root->updateMBR();
            root = new_root;
        } else {
            // Add the new node to the parent
            parent->children.push_back(new_node);
            new_node->parent = parent;
            parent->updateMBR();
            
            // Check if the parent needs to be split
            if (parent->children.size() > max_entries) {
                splitNode(parent);
            }
        }
    }
    
    void splitNonLeafNode(std::shared_ptr<Node> node) {
        // Create a new node
        auto new_node = std::make_shared<Node>(false);
        
        // Find the seeds for splitting
        int seed1 = 0, seed2 = 0;
        findSeedsNonLeaf(node->children, seed1, seed2);
        
        // Create two groups
        std::vector<std::shared_ptr<Node>> group1, group2;
        group1.push_back(node->children[seed1]);
        group2.push_back(node->children[seed2]);
        
        // Create a list of children to distribute
        std::vector<std::shared_ptr<Node>> remaining;
        for (size_t i = 0; i < node->children.size(); i++) {
            if (i != seed1 && i != seed2) {
                remaining.push_back(node->children[i]);
            }
        }
        
        // Calculate initial MBRs
        Rectangle mbr1 = group1[0]->mbr;
        Rectangle mbr2 = group2[0]->mbr;
        
        // Distribute remaining children
        while (!remaining.empty()) {
            // If one group has too few entries, assign all remaining to it
            if (group1.size() + remaining.size() <= min_entries) {
                group1.insert(group1.end(), remaining.begin(), remaining.end());
                break;
            }
            if (group2.size() + remaining.size() <= min_entries) {
                group2.insert(group2.end(), remaining.begin(), remaining.end());
                break;
            }
            
            // Find the child that has the maximum difference in area increase
            int selected = 0;
            double max_diff = -std::numeric_limits<double>::max();
            int preferred_group = 0;
            
            for (size_t i = 0; i < remaining.size(); i++) {
                Rectangle extended1 = mbr1;
                extended1.extend(remaining[i]->mbr);
                double increase1 = extended1.area() - mbr1.area();
                
                Rectangle extended2 = mbr2;
                extended2.extend(remaining[i]->mbr);
                double increase2 = extended2.area() - mbr2.area();
                
                double diff = std::abs(increase1 - increase2);
                
                if (diff > max_diff) {
                    max_diff = diff;
                    selected = i;
                    preferred_group = (increase1 < increase2) ? 1 : 2;
                }
            }
            
            // Assign the selected child to the preferred group
            if (preferred_group == 1) {
                group1.push_back(remaining[selected]);
                mbr1.extend(remaining[selected]->mbr);
            } else {
                group2.push_back(remaining[selected]);
                mbr2.extend(remaining[selected]->mbr);
            }
            
            // Remove the selected child from the remaining list
            remaining.erase(remaining.begin() + selected);
        }
        
        // Update the nodes
        node->children = group1;
        new_node->children = group2;
        
        // Update parent pointers for all children
        for (auto& child : node->children) {
            child->parent = node;
        }
        for (auto& child : new_node->children) {
            child->parent = new_node;
        }
        
        // Update MBRs
        node->updateMBR();
        new_node->updateMBR();
        
        // Handle the parent
        auto parent = node->parent.lock();
        if (!parent) {
            // This is the root, create a new root
            auto new_root = std::make_shared<Node>(false);
            new_root->children.push_back(node);
            new_root->children.push_back(new_node);
            node->parent = new_root;
            new_node->parent = new_root;
            new_root->updateMBR();
            root = new_root;
        } else {
            // Add the new node to the parent
            parent->children.push_back(new_node);
            new_node->parent = parent;
            parent->updateMBR();
            
            // Check if the parent needs to be split
            if (parent->children.size() > max_entries) {
                splitNode(parent);
            }
        }
    }
    
    template <typename T>
    void findSeeds(const std::vector<T>& items, int& seed1, int& seed2) {
        double max_dist = -1;
        
        for (size_t i = 0; i < items.size(); i++) {
            for (size_t j = i + 1; j < items.size(); j++) {
                Rectangle mbr_i, mbr_j;
                
                if constexpr (std::is_same_v<T, Entry>) {
                    mbr_i = items[i].mbr;
                    mbr_j = items[j].mbr;
                } else {
                    mbr_i = items[i]->mbr;
                    mbr_j = items[j]->mbr;
                }
                
                double dist = 0;
                for (int d = 0; d < DIMENSIONS; d++) {
                    double diff_min = mbr_i.min[d] - mbr_j.min[d];
                    double diff_max = mbr_i.max[d] - mbr_j.max[d];
                    dist += diff_min * diff_min + diff_max * diff_max;
                }
                
                if (dist > max_dist) {
                    max_dist = dist;
                    seed1 = i;
                    seed2 = j;
                }
            }
        }
    }
    
    void findSeedsNonLeaf(const std::vector<std::shared_ptr<Node>>& children, int& seed1, int& seed2) {
        findSeeds(children, seed1, seed2);
    }

public:
    RTree(int max_entries = 10) : max_entries(max_entries) {
        min_entries = max_entries / 2;
        root = std::make_shared<Node>(true);
    }
    
    void insert(int id, const std::vector<double>& point) {
        // Store the data point
        data_points[id] = point;
        
        // Create an MBR for this point
        Rectangle mbr(point);
        
        // Create an entry
        Entry entry(mbr, id);
        
        // Choose a leaf node
        auto leaf = chooseLeaf(root, mbr);
        
        // Add the entry to the leaf
        leaf->entries.push_back(entry);
        
        // Adjust the tree
        adjustTree(leaf);
        
        // Check if the leaf needs to be split
        if (leaf->entries.size() > max_entries) {
            splitNode(leaf);
        }
    }
    
    bool remove(int id) {
        // Find the leaf node containing the entry
        auto leaf = findLeaf(root, id);
        if (!leaf) return false;
        
        // Find the entry in the leaf
        auto it = std::find_if(leaf->entries.begin(), leaf->entries.end(),
                               [id](const Entry& e) { return e.data_id == id; });
        if (it == leaf->entries.end()) return false;
        
        // Remove the entry
        leaf->entries.erase(it);
        
        // Adjust the tree
        adjustTree(leaf);
        
        // Remove the data point
        data_points.erase(id);
        
        // Check for underflow and condense if needed
        condenseTree(leaf);
        
        // Check if the root has only one child and is not a leaf
        if (!root->is_leaf && root->children.size() == 1) {
            auto new_root = root->children[0];
            new_root->parent.reset();
            root = new_root;
        }
        
        return true;
    }
    
    std::shared_ptr<Node> findLeaf(std::shared_ptr<Node> node, int id) {
        if (!node) return nullptr;
        
        if (node->is_leaf) {
            for (const auto& entry : node->entries) {
                if (entry.data_id == id) {
                    return node;
                }
            }
            return nullptr;
        }
        
        for (auto& child : node->children) {
            auto found = findLeaf(child, id);
            if (found) return found;
        }
        
        return nullptr;
    }
    
    void condenseTree(std::shared_ptr<Node> node) {
        std::vector<std::shared_ptr<Node>> deleted_nodes;
        
        auto current = node;
        while (current != root) {
            auto parent = current->parent.lock();
            if (!parent) break;
            
            // Check for underflow
            if ((current->is_leaf && current->entries.size() < min_entries) ||
                (!current->is_leaf && current->children.size() < min_entries)) {
                // Remove from parent
                auto it = std::find(parent->children.begin(), parent->children.end(), current);
                if (it != parent->children.end()) {
                    parent->children.erase(it);
                }
                
                // Add to deleted nodes
                deleted_nodes.push_back(current);
            }
            
            current = parent;
        }
        
        // Reinsert orphaned entries
        for (auto& node : deleted_nodes) {
            if (node->is_leaf) {
                for (auto& entry : node->entries) {
                    insert(entry.data_id, data_points[entry.data_id]);
                }
            } else {
                for (auto& child : node->children) {
                    reinsertNode(child);
                }
            }
        }
    }
    
    void reinsertNode(std::shared_ptr<Node> node) {
        if (node->is_leaf) {
            for (auto& entry : node->entries) {
                insert(entry.data_id, data_points[entry.data_id]);
            }
        } else {
            for (auto& child : node->children) {
                reinsertNode(child);
            }
        }
    }
    
    std::vector<int> search(const Rectangle& query_rect) {
        std::vector<int> result;
        search(root, query_rect, result);
        return result;
    }
    
    void search(std::shared_ptr<Node> node, const Rectangle& query_rect, std::vector<int>& result) {
        if (!node) return;
        
        if (!node->mbr.intersects(query_rect)) {
            return;
        }
        
        if (node->is_leaf) {
            for (const auto& entry : node->entries) {
                if (entry.mbr.intersects(query_rect)) {
                    result.push_back(entry.data_id);
                }
            }
        } else {
            for (auto& child : node->children) {
                search(child, query_rect, result);
            }
        }
    }
    
    // Corrected kNearestNeighbors implementation
    std::vector<std::pair<int, double>> kNearestNeighbors(const std::vector<double>& query_point, int k) {
        // Max heap for k nearest neighbors
        std::priority_queue<std::pair<double, int>> result_heap;
        
        // Min heap for branch-and-bound search (nodes sorted by min distance)
        std::priority_queue<std::pair<double, std::shared_ptr<Node>>, 
                          std::vector<std::pair<double, std::shared_ptr<Node>>>,
                          std::greater<std::pair<double, std::shared_ptr<Node>>>> node_queue;
        
        // Add the root to the queue
        node_queue.push({root->mbr.minDist(query_point), root});
        
        // Current kth distance (initialize to infinity)
        double kth_dist = std::numeric_limits<double>::max();
        
        // Process nodes in order of increasing distance
        while (!node_queue.empty()) {
            auto [min_dist, node] = node_queue.top();
            node_queue.pop();
            
            // If we've found k points and the minimum distance to this node
            // is greater than the kth nearest distance, we can stop
            if (min_dist > kth_dist && result_heap.size() >= k) {
                break;
            }
            
            if (node->is_leaf) {
                for (const auto& entry : node->entries) {
                    double distance = 0;
                    const auto& point = data_points[entry.data_id];
                    for (int i = 0; i < DIMENSIONS; i++) {
                        double diff = query_point[i] - point[i];
                        distance += diff * diff;
                    }
                    
                    result_heap.push({distance, entry.data_id});
                    
                    // If we have more than k results, remove the farthest one
                    if (result_heap.size() > k) {
                        result_heap.pop();
                    }
                    
                    // Update kth distance if we have k results
                    if (result_heap.size() == k) {
                        kth_dist = result_heap.top().first;
                    }
                }
            } else {
                // Add all children to the queue
                for (auto& child : node->children) {
                    double min_dist = child->mbr.minDist(query_point);
                    node_queue.push({min_dist, child});
                }
            }
        }
        
        // Convert priority queue to vector and sort by distance
        std::vector<std::pair<int, double>> result;
        while (!result_heap.empty()) {
            auto [dist, id] = result_heap.top();
            result_heap.pop();
            result.push_back({id, dist});
        }
        
        // Reverse to get ascending order by distance
        std::reverse(result.begin(), result.end());
        
        return result;
    }
    
    // Serialization methods
    void save(const std::string& filename) {
        std::ofstream out(filename, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Could not open file for writing");
        }
        
        // Save configuration
        out.write(reinterpret_cast<const char*>(&max_entries), sizeof(int));
        out.write(reinterpret_cast<const char*>(&min_entries), sizeof(int));
        
        // Save root
        root->serialize(out);
        
        // Save data points
        size_t points_size = data_points.size();
        out.write(reinterpret_cast<const char*>(&points_size), sizeof(size_t));
        
        for (const auto& [id, point] : data_points) {
            out.write(reinterpret_cast<const char*>(&id), sizeof(int));
            for (int i = 0; i < DIMENSIONS; i++) {
                out.write(reinterpret_cast<const char*>(&point[i]), sizeof(double));
            }
        }
        
        out.close();
    }
    
    void load(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Could not open file for reading");
        }
        
        // Load configuration
        in.read(reinterpret_cast<char*>(&max_entries), sizeof(int));
        in.read(reinterpret_cast<char*>(&min_entries), sizeof(int));
        
        // Load root
        root = std::make_shared<Node>(false);
        root->deserialize(in, root);
        
        // Load data points
        size_t points_size;
        in.read(reinterpret_cast<char*>(&points_size), sizeof(size_t));
        
        data_points.clear();
        for (size_t i = 0; i < points_size; i++) {
            int id;
            in.read(reinterpret_cast<char*>(&id), sizeof(int));
            
            std::vector<double> point(DIMENSIONS);
            for (int j = 0; j < DIMENSIONS; j++) {
                in.read(reinterpret_cast<char*>(&point[j]), sizeof(double));
            }
            
            data_points[id] = point;
        }
        
        in.close();
    }
    
    void clear() {
        root = std::make_shared<Node>(true);
        data_points.clear();
    }
    
    size_t size() const {
        return data_points.size();
    }
    
    bool empty() const {
        return data_points.empty();
    }
    
    int getMaxEntries() const {
        return max_entries;
    }
    
    int getMinEntries() const {
        return min_entries;
    }
    
    const std::shared_ptr<Node>& getRoot() const {
        return root;
    }
};

#endif // RTREE_H