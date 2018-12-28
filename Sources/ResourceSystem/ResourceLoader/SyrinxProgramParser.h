#pragma once
#include <string>
#include <vector>
#include <utility>
#include "ResourceManager/SyrinxFileManager.h"

namespace Syrinx {

class ProgramParser {
public:
    ProgramParser(const std::string& fileName, FileManager *fileManager) noexcept(false);
    const std::string& getSource() const;
    const std::vector<std::string>& getIncludedFileList() const;

private:
    std::string parseFile(const std::string& fileName);
    std::pair<bool, std::string> parseLine(const std::string& line) const;
    size_t ignoreBlankCharacter(const std::string& line, size_t start) const;

private:
    std::string mFileName;
    std::vector<std::string> mIncludedFileList;
    std::string mSource;
    FileManager *mFileManager;
};

} // namespace Syrinx
