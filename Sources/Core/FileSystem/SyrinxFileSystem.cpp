#include "FileSystem/SyrinxFileSystem.h"
#include <filesystem>
#include "Common/SyrinxAssert.h"
#include "Exception/SyrinxException.h"
#include "Logging/SyrinxLogManager.h"

namespace Syrinx {

namespace fs = std::filesystem;

void FileSystem::setWorkingDirectory(const std::string& path)
{
    SYRINX_EXPECT(!path.empty());
    try {
        fs::current_path(path);
    } catch (std::exception& e) {
        SYRINX_ERROR_FMT("fail to set working directory to path [{}] because [{}]", path, e.what());
        throw;
    }
}


std::string FileSystem::getWorkingDirectory()
{
    try {
        return fs::current_path().string();
    } catch (std::exception& e) {
        SYRINX_ERROR_FMT("fail to get working directory because [{}]", e.what());
        throw;
    }
}


bool FileSystem::fileExist(const std::string& path)
{
    SYRINX_EXPECT(!path.empty());
    try {
        return fs::is_regular_file(path);
    } catch (std::exception& e) {
        SYRINX_ERROR_FMT("fail to check if the path [{}] corresponds to an existing file, because [{}]", path, e.what());
        throw;
    }
}


std::pair<bool, std::string> FileSystem::findFileRecursivelyInDirectory(const std::string& fileName, const std::string& directoryPath)
{
    if (!directoryExist(directoryPath)) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "path [{}] is not a directory", directoryPath);
    }

    try {
        fs::path filePath = fs::path(directoryPath) / fileName;
        if (fileExist(filePath.string())) {
            return {true, weaklyCanonical(filePath.string())};
        }

        for (auto& path : fs::recursive_directory_iterator(directoryPath)) {
            filePath = path / fileName;
            if (fileExist(filePath.string())) {
                return {true, weaklyCanonical(filePath.string())};
            }
        }
        return {false, ""};
    } catch (std::exception& e) {
        SYRINX_ERROR_FMT("fail to find file [{}] in directory [{}] because {}", fileName, directoryPath, e.what());
        throw;
    }
}


bool FileSystem::directoryExist(const std::string& path)
{
    SYRINX_EXPECT(!path.empty());
    try {
        return fs::is_directory(path);
    } catch (std::exception& e) {
        SYRINX_ERROR_FMT("fail to check if the path [{}] corresponds to an existing directory, because [{}]", path, e.what());
        throw;
    }
}


void FileSystem::createDirectory(const std::string& path)
{
    SYRINX_EXPECT(!path.empty());
    try {
        fs::create_directory(path);
    } catch (std::exception& e) {
        SYRINX_ERROR_FMT("fail to create directory [{}] because of [{}]", path, e.what());
        throw;
    }
}


void FileSystem::remove(const std::string& path)
{
    try {
        fs::remove(path);
    } catch (std::exception& e) {
        SYRINX_ERROR_FMT("fail to remove path [{}] because [{}]", path, e.what());
    }
}


std::string FileSystem::combine(const std::string& root, const std::string& relative)
{
    try {
        fs::path rootPath(root);
        fs::path relativePath(relative);

        if (!rootPath.is_absolute()) {
            SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileSystemError, "path [{}] is not absolute", root);
        }

        if (!relativePath.is_relative()) {
            SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileSystemError, "path [{}] is not relative path", relative);
        }

        auto combinedPath = fs::path(root) / relativePath;
        return fs::weakly_canonical(combinedPath).string();
    } catch (std::exception& e) {
        SYRINX_ERROR_FMT("fail to combine [{}] and [{}], because [{}]", root, relative, e.what());
        throw;
    }
}


std::string FileSystem::canonical(const std::string& path)
{
    SYRINX_EXPECT(!path.empty());
    try {
        return fs::canonical(path).string();
    } catch (std::exception& e) {
        SYRINX_ERROR_FMT("fail to get canonical path for path [{}] because of [{}]", path, e.what());
        throw;
    }
}


std::string FileSystem::weaklyCanonical(const std::string& path)
{
    SYRINX_EXPECT(!path.empty());
    try {
        return fs::weakly_canonical(path).string();
    } catch (std::exception& e) {
        SYRINX_ERROR_FMT("fail to get weakly canonical path for path [{}] because of [{}]", path, e.what());
        throw;
    }
}


std::string FileSystem::getParentPath(const std::string& path)
{
    SYRINX_EXPECT(!path.empty());
    return fs::path(path).parent_path().string();
}


std::string FileSystem::getFileName(const std::string& path)
{
    SYRINX_EXPECT(!path.empty());
    return fs::path(path).filename().string();
}


FileSystem::FileTime FileSystem::getLastWriteTime(const std::string& path)
{
    SYRINX_EXPECT(!path.empty());
    if (!fileExist(path)) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileNotFound ,"file [{}] doesn't exists", path);
    }
    return fs::last_write_time(path);
}

} // namespace Syrinx