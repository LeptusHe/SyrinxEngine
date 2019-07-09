#pragma once
#include <vector>
#include <shaderc/shaderc.hpp>
#include <ResourceManager/SyrinxFileManager.h>

namespace Syrinx {

class FileIncluder : public shaderc::CompileOptions::IncluderInterface {
public:
    FileIncluder() = default;
    explicit FileIncluder(const std::vector<std::string>& searchPathList);
    ~FileIncluder() override = default;

    void addSearchPath(const std::string& searchPath);
    shaderc_include_result* GetInclude(const char* requestedSource,
                                       shaderc_include_type includeType,
                                       const char* requestingSource,
                                       size_t includeDepth) override;
    void ReleaseInclude(shaderc_include_result* includeResult) override;

private:
    struct FileInfo {
        const std::string fullPath;
        std::string mContent;
    };

private:
    FileManager mFileManager;
    std::vector<std::string> mIncludedFileList;
};

} // namespace Syrinx