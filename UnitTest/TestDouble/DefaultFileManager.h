#pragma once
#include <unordered_map>
#include <ResourceManager/SyrinxFileManager.h>
#include <Streaming/SyrinxDataStream.h>


class DefaultFileManager : public Syrinx::FileManager {
public:
    ~DefaultFileManager() override
    {
        for (const auto& [name, fileStream] : mFileStreamMap) {
            delete fileStream;
        }
    }


    std::pair<bool, std::string> findFile(const std::string& fileName) const override
    {
        auto iter = mFileStreamMap.find(fileName);
        if (iter != std::end(mFileStreamMap)) {
            return {true, fileName};
        }
        return {false, ""};
    }


    Syrinx::DataStream* openFile(const std::string& fileName, Syrinx::FileAccessMode accessMode) const override
    {
        auto iter = mFileStreamMap.find(fileName);
        if (iter != std::end(mFileStreamMap)) {
            return iter->second;
        }
        return nullptr;
    }


    void addFileStream(Syrinx::DataStream* fileStream)
    {
        SYRINX_EXPECT(fileStream);
        mFileStreamMap[fileStream->getName()] = fileStream;
    }

private:
    std::unordered_map<std::string, Syrinx::DataStream*> mFileStreamMap;
};
