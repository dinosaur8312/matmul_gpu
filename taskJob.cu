#include "taskJob.h"
#include <fstream>
#include <sstream>
#include <iostream>

// Parse MIL file and create TaskBins
void TaskJob::parseMILFile(const std::string &filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filePath << std::endl;
        return;
    }

    std::string line;
    std::vector<Task> tasks; // Temporary storage for tasks

    while (std::getline(file, line)) {
        std::istringstream lineStream(line);
        std::string word;
        std::vector<std::string> words;
        while (lineStream >> word) {
            words.push_back(word);
        }

        // Check if the line has at least 7 elements (based on the given format)
        if (words.size() >= 7) {
            int M = std::stoi(words[4]);
            int N = std::stoi(words[5]);
            int R = std::stoi(words[6]);
            tasks.emplace_back(M, N, R);
        }
    }

    file.close();

    // Allocate memory for each task and group them into TaskBins
    for (auto &task : tasks) {
        task.allocateDeviceMemory(10); // Example nRHS = 10
    }

    // Group tasks into bins
    std::map<std::tuple<int, int, int>, TaskBin> binsMap;
    for (const auto &task : tasks) {
        std::tuple<int, int, int> binKey = std::make_tuple(task.M_pad, task.N_pad, task.R_pad);

        if (binsMap.find(binKey) == binsMap.end()) {
            binsMap.emplace(binKey, TaskBin(task.M_pad, task.N_pad, task.R_pad));
        }
        binsMap[binKey].addTask(task);
    }

    // Transfer bins from map to vector
    taskBins.clear();
    for (auto &[key, bin] : binsMap) {
        taskBins.push_back(std::move(bin));
    }
}
