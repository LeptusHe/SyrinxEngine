#pragma once
#include <FileSystem/SyrinxFileManager.h>
#include <Manager/SyrinxHardwareResourceManager.h>
#include <Program/SyrinxProgramCompiler.h>
#include "RenderResource/SyrinxShader.h"
#include "SyrinxResourceManager.h"
#include "SyrinxShaderFileIncluder.h"

namespace Syrinx {

class ShaderManager : public ResourceManager<Shader> {
public:
    ShaderManager(FileManager *fileManager, HardwareResourceManager *hardwareResourceManager);
    ~ShaderManager() override = default;

    void addShaderSearchPath(const std::string& path);
    std::unique_ptr<Shader> create(const std::string& name) override;
    virtual FileManager* getFileManager() const;

private:
    FileManager *mFileManager;
    HardwareResourceManager *mHardwareResourceManager;
    ProgramCompiler mCompiler;
    std::vector<std::string> mSearchPathList;
};

} // namespace Syrinx