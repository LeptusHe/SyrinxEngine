#pragma once
#include <FileSystem/SyrinxFileSystem.h>

namespace Syrinx {

class FileDialog {
public:
    static FileDialog& getInstance();
    std::pair<bool, std::string> open(const std::string& title, float width, float height, const std::string& directory = "");

private:
    FileDialog() = default;
    ~FileDialog() = default;
    std::vector<std::string> splitDirectory(const std::string& directory) const;
    std::string getSubDirectory(const std::string& lastFileName);
    void setDirectory(const std::string& directory);

private:
    std::string mDirectory;
    std::string mSelectedEntry;
    FileSystem mFileSystem;
};

} // namespace Syrinx