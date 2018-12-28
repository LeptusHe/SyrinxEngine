#include "FileSystem/SyrinxFileWatcher.h"
#include "Common/SyrinxAssert.h"
#include "Exception/SyrinxException.h"
#include "Logging/SyrinxLogManager.h"

namespace Syrinx {

FileWatcher::FileWatcher(const std::string& filePath, std::unique_ptr<Syrinx::FileSystem>&& fileSystem)
    : mFilePath(filePath)
    , mIsWatching(false)
    , mLastWriteTime()
    , mListenerList()
    , mFileSystem(std::move(fileSystem))
{
    SYRINX_ENSURE(!mFilePath.empty());
    SYRINX_ENSURE(mFilePath == filePath);
    SYRINX_ENSURE(!mIsWatching);
    SYRINX_ENSURE(mLastWriteTime.time_since_epoch().count() == 0);
    SYRINX_ENSURE(mListenerList.empty());
    SYRINX_ENSURE(mFileSystem);
    SYRINX_ENSURE(!fileSystem);

    mFilePath = mFileSystem->weaklyCanonical(filePath);
    if (!mFileSystem->fileExist(mFilePath)) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileNotFound, "fail to find file [{}]", mFilePath);
    }
}


void FileWatcher::addListener(const FileWatcher::Listener& listener)
{
    mListenerList.push_back(listener);
}


const std::string& FileWatcher::getFilePath() const
{
    return mFilePath;
}


const FileWatcher::ListenerList& FileWatcher::getListenerList() const
{
    return mListenerList;
}


bool FileWatcher::isWatching() const
{
    return mIsWatching;
}


void FileWatcher::startToWatch()
{
    SYRINX_EXPECT(!isWatching());
    mIsWatching = true;
    mLastWriteTime = mFileSystem->getLastWriteTime(mFilePath);
    SYRINX_ENSURE(isWatching());
}


void FileWatcher::stopWatching()
{
    SYRINX_EXPECT(isWatching());
    mIsWatching = false;
    SYRINX_ENSURE(!isWatching());
}



bool FileWatcher::isModified()
{
    if (!isWatching()) {
        return false;
    }

    FileTime lastWriteTime = mFileSystem->getLastWriteTime(mFilePath);
    bool isModified = lastWriteTime.time_since_epoch().count() > mLastWriteTime.time_since_epoch().count();
    if (isModified) {
        mLastWriteTime = lastWriteTime;
        std::time_t time = std::chrono::system_clock::to_time_t(mLastWriteTime);
        SYRINX_INFO_FMT("file [{}] is modified in time [{}]", getFilePath(), std::ctime(&time));
    }
    return isModified;
}

} // namespace Syrinx