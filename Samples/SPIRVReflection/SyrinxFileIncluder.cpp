#include "SyrinxFileIncluder.h"
#include <Common/SyrinxAssert.h>
#include <Logging/SyrinxLogManager.h>

namespace Syrinx {

shaderc_include_result* MakeErrorIncludeResult(const char* message)
{
    return new shaderc_include_result{"", 0, message, std::strlen(message)};
}


FileIncluder::FileIncluder(const std::vector<std::string>& searchPathList)
{
    for (const auto& searchPath : searchPathList) {
        mFileManager.addSearchPath(searchPath);
    }
}


void FileIncluder::addSearchPath(const std::string& searchPath)
{
    mFileManager.addSearchPath(searchPath);
}


shaderc_include_result* FileIncluder::GetInclude(const char *requestedSource, shaderc_include_type includeType,
                                                 const char *requestingSource, size_t includeDepth)
{
    SYRINX_EXPECT(requestedSource && requestingSource);

    std::string fullPath;
    if (includeType == shaderc_include_type_relative) {
        auto fileSystem = mFileManager.getFileSystem();
        auto basePath = fileSystem->getParentPath(requestingSource);
        fullPath = fileSystem->combine(basePath, requestedSource);
    } else {
        auto [exists, filePath] = mFileManager.findFile(requestedSource);
        if (!exists) {
            SYRINX_INFO_FMT("fail to find include file [{}] in file [{}]", requestedSource, requestingSource);
            return MakeErrorIncludeResult("cannot find or open include file");
        }
        fullPath = filePath;
    }

    auto fileStream = mFileManager.openFile(fullPath, FileAccessMode::READ);
    if (!fileStream) {
        SYRINX_INFO_FMT("fail to find include file [{}] in file [{}]", fullPath, requestingSource);
        return MakeErrorIncludeResult("cannot find or open include file");
    }
    FileInfo *fileInfo = new FileInfo{fullPath, fileStream->getAsString()};
    return new shaderc_include_result{
        fileInfo->fullPath.c_str(), fileInfo->fullPath.length(),
        fileInfo->mContent.data(), fileInfo->mContent.size(),
        fileInfo
    };
}


void FileIncluder::ReleaseInclude(shaderc_include_result *includeResult)
{
    SYRINX_EXPECT(includeResult);
    auto *fileInfo = reinterpret_cast<FileInfo*>(includeResult->user_data);
    delete fileInfo;
    delete includeResult;
}

} // namespace Syrinx
