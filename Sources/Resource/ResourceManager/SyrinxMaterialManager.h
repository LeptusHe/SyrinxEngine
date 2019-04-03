#pragma once
#include <unordered_map>
#include "RenderResource/SyrinxMaterial.h"
#include "ResourceManager/SyrinxFileManager.h"
#include "ResourceManager/SyrinxResourceManager.h"
#include "ResourceManager/SyrinxHardwareResourceManager.h"

namespace Syrinx {

class MaterialManager : public ResourceManager<Material> {
public:
    using ShaderMap = std::unordered_map<std::string, std::unique_ptr<Shader>>;
    using ShaderList = std::vector<Shader*>;

public:
    MaterialManager(FileManager *fileManager, HardwareResourceManager *hardwareResourceManager);
    ~MaterialManager() override = default;

    std::unique_ptr<Material> create(const std::string& name) override;
    virtual FileManager* getFileManager() const;

    virtual Shader* createShader(const std::string& name);
    virtual Shader* findShader(const std::string& name) const;

private:
    void addShader(Shader *shader);

private:
    FileManager *mFileManager;
    HardwareResourceManager *mHardwareResourceManager;
    ShaderMap mShaderMap;
    ShaderList mShaderList;
};

} // namespace Syrinx

