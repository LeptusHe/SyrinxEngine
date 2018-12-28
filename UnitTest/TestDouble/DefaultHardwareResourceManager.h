#include <ResourceManager/SyrinxHardwareResourceManager.h>
#include <unordered_map>
#include <TestDouble/DefaultProgramStage.h>

class DefaultHardwareResourceManager : public Syrinx::HardwareResourceManager {
public:
    explicit DefaultHardwareResourceManager(Syrinx::FileManager *fileManager) : HardwareResourceManager(fileManager) { }
    ~DefaultHardwareResourceManager() override = default;

public:
    Syrinx::ProgramStage* createProgramStage(const std::string& fileName, Syrinx::ProgramStageType stageType) override
    {
        auto program = new DefaultProgramStage(fileName);
        addProgramStage(program);
        return program;
    }
};