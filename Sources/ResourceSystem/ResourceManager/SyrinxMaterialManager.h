#pragma once
#include <unordered_map>
#include "RenderResource/SyrinxMaterial.h"
#include "ResourceManager/SyrinxFileManager.h"
#include "ResourceManager/SyrinxHardwareResourceManager.h"

namespace Syrinx {

class MaterialManager {
public:
    using MaterialMap = std::unordered_map<std::string, std::unique_ptr<Material>>;
    using MaterialList = std::vector<Material*>;
    using ShaderMap = std::unordered_map<std::string, std::unique_ptr<Shader>>;
    using ShaderList = std::vector<Shader*>;

public:
    explicit MaterialManager(FileManager *fileManager, HardwareResourceManager *hardwareResourceManager);
    virtual ~MaterialManager() = default;
    virtual Material* createMaterial(const std::string& name);
    virtual Material* findMaterial(const std::string& name) const;
    virtual Shader* createShader(const std::string& name);
    virtual Shader* findShader(const std::string& name) const;

private:
    virtual Material* parseMaterial(const std::string& filePath);
    virtual void addMaterial(Material *material);
    virtual void addShader(Shader *shader);

private:
    FileManager *mFileManager;
    HardwareResourceManager *mHardwareResourceManager;
    MaterialMap mMaterialMap;
    MaterialList mMaterialList;
    ShaderMap mShaderMap;
    ShaderList mShaderList;
};

} // namespace Syrinx

