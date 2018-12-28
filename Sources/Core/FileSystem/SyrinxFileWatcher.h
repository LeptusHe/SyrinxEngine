#pragma once
#include <filesystem>
#include <functional>
#include "FileSystem/SyrinxFileSystem.h"

namespace Syrinx {

class FileWatcher {
public:
    using Listener = std::function<void(const std::string&)>;
    using ListenerList = std::vector<Listener>;
    using FileTime = FileSystem::FileTime;

public:
    explicit FileWatcher(const std::string& filePath, std::unique_ptr<FileSystem>&& fileSystem = std::make_unique<FileSystem>()) noexcept(false);
    void addListener(const Listener& listener);
    const std::string& getFilePath() const;
    const ListenerList& getListenerList() const;
    bool isWatching() const;
    void startToWatch();
    void stopWatching();
    bool isModified();

private:
    std::string mFilePath;
    bool mIsWatching;
    FileTime mLastWriteTime;
    ListenerList mListenerList;
    std::unique_ptr<FileSystem> mFileSystem;
};

} // namespace Syrinx