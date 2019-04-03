#pragma once
#include "RenderResource/SyrinxShader.h"
#include "ResourceManager/SyrinxFileManager.h"
#include "ResourceManager/SyrinxResourceManager.h"
#include "ResourceManager/SyrinxHardwareResourceManager.h"

namespace Syrinx {

class ShaderManager : public ResourceManager<Shader> {
public:
    ShaderManager(FileManager *fileManager, HardwareResourceManager *hardwareResourceManager);
    ~ShaderManager() override = default;

    std::unique_ptr<Shader> create(const std::string& name) override;
    virtual FileManager* getFileManager() const;

private:
    FileManager *mFileManager;
    HardwareResourceManager *mHardwareResourceManager;
};

} // namespace Syrinx