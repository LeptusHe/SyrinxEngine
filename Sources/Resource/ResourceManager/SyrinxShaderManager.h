#pragma once
#include "RenderResource/SyrinxShader.h"
#include "ResourceManager/SyrinxFileManager.h"
#include "ResourceManager/SyrinxResourceManager.h"

namespace Syrinx {

class ShaderManager : public ResourceManager<Shader> {
public:
    explicit ShaderManager(FileManager *fileManager);
    ~ShaderManager() override = default;

    std::unique_ptr<Shader> create(const std::string& name) override;
    virtual FileManager* getFileManager() const;

private:
    FileManager *mFileManager;
};

} // namespace Syrinx