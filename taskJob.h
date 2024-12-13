#ifndef TASKJOB_H
#define TASKJOB_H

#include "taskBin.h"
#include <string>
#include <vector>
#include <map>

class TaskJob {
private:
    std::vector<TaskBin> taskBins; // Vector of TaskBins
public:
    TaskJob() = default;
    ~TaskJob() = default;

    // Parse MIL file and create TaskBins
    void parseMILFile(const std::string &filePath);

    // Get TaskBins
    const std::vector<TaskBin> &getTaskBins() const { return taskBins; }
};

#endif // TASKJOB_H
