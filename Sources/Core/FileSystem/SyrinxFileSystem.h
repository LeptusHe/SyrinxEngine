#pragma once
#include <string>
#include <vector>
#include <utility>
#include <filesystem>

namespace Syrinx {

class FileSystem {
public:
	using FileTime = std::filesystem::file_time_type;
	using Path = std::filesystem::path;

public:
	FileSystem() = default;
	~FileSystem() = default;

public:
    virtual void setWorkingDirectory(const std::string& path) noexcept(false);
    virtual std::string getWorkingDirectory() noexcept(false);
    virtual bool fileExist(const std::string& path) noexcept(false);
    virtual std::pair<bool, std::string> findFileRecursivelyInDirectory(const std::string& fileName, const std::string& directoryPath) noexcept(false);
    virtual bool directoryExist(const std::string& path) noexcept(false);
	virtual void createDirectory(const std::string& path) noexcept(false);
	virtual void remove(const std::string& path) noexcept(false);
	virtual std::string combine(const std::string& root, const std::string& relative) noexcept(false);
	virtual std::string canonical(const std::string& path) noexcept(false);
	virtual std::string weaklyCanonical(const std::string& path) noexcept(false);
	virtual std::string getParentPath(const std::string& path) noexcept(false);
	virtual std::string getFileName(const std::string& path) noexcept(false);
	virtual FileTime getLastWriteTime(const std::string& path) noexcept(false);
	virtual std::vector<std::string> getEntryListInDirectory(const std::string& path) noexcept(false);
};

} // namespace Syrinx