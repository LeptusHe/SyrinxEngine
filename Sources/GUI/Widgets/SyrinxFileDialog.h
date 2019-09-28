#pragma once
#include <FileSystem/SyrinxFileSystem.h>

namespace Syrinx {

class FileDialog {
public:
    static FileDialog& getInstance();
    bool open(const std::string& title, const std::string& directory = "");

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