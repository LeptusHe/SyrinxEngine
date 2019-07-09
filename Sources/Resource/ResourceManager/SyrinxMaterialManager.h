#pragma once
#include <FileSystem/SyrinxFileManager.h>
#include "RenderResource/SyrinxMaterial.h"
#include "ResourceManager/SyrinxResourceManager.h"
#include <Manager/SyrinxHardwareResourceManager.h>
#include "ResourceManager/SyrinxShaderManager.h"

namespace Syrinx {

class MaterialManager : public ResourceManager<Material> {
public:
    MaterialManager(FileManager *fileManager, HardwareResourceManager *hardwareResourceManager, ShaderManager *shaderManager);
    ~MaterialManager() override = default;

    std::unique_ptr<Material> create(const std::string& name) override;
    virtual FileManager* getFileManager() const;

private:
    FileManager *mFileManager;
    HardwareResourceManager *mHardwareResourceManager;
    ShaderManager *mShaderManager;
};

} // namespace Syrinx

