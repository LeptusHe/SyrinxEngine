#include <Script/SyrinxLuaCommon.h>
#include <Logging/SyrinxLogManager.h>
#include <FileSystem/SyrinxFileManager.h>
#include <Program/SyrinxProgramVariables.h>
#include <Manager/SyrinxHardwareResourceManager.h>
#include <Program/SyrinxProgramCompiler.h>

void parseProgramParameters(const sol::table& parameters)
{
}


struct Vec3 {
    Vec3(float x, float y, float z) : mX(x), mY(y), mZ(z) {}
    float mX, mY, mZ;
};

int main(int argc, char *argv[])
{
    auto logger = std::make_unique<Syrinx::LogManager>();
    Syrinx::FileManager fileManager;

    fileManager.addSearchPath(".");
    auto [fileExist, configFilePath] = fileManager.findFile("config.lua");
    if (!fileExist) {
        return -1;
    }

    Syrinx::HardwareResourceManager resourceManager;
    Syrinx::ProgramCompiler compiler;

    fileManager.addSearchPath(".");
    auto fileStream = fileManager.openFile("VertexShader.vert", Syrinx::FileAccessMode::READ);
    SYRINX_ASSERT(fileStream);

    auto binarySource = compiler.compile("VertexShader.vert", fileStream->getAsString(), Syrinx::ProgramStageType::VertexStage);
    auto programPtr = resourceManager.createProgramStage("vertex", std::move(binarySource), Syrinx::ProgramStageType::VertexStage);

    auto programVarsPtr = programPtr->getProgramVars();
    auto& programVars = *(programVarsPtr);



    auto structConstructor = [&](const sol::table& parameters) {
        for (const auto& kvPair : parameters) {
            const auto& key = kvPair.first;
            const auto& value = kvPair.second;
            if (value.get_type() == sol::type::table) {

            } else {
                std::string paramName = key.as<std::string>();
                programVars.[paramName] = value.as<float>();
            }
        }
    };

    sol::state lua;
    lua.new_usertype<glm::vec3>("vec3",
        sol::constructors<glm::vec3(), glm::vec3(float), glm::vec3(float, float, float)>());
    lua["Struct"] = structConstructor;

    lua.script_file(configFilePath);


    //Vec3 value = lua["test"];
    //std::cout << value.mX << ", " << value.mY << ", " << value.mZ << std::endl;

    /*
    sol::table passTable = lua["pass"];

    for (const auto& kvPair : passTable) {
        sol::object key = kvPair.first;
        sol::object value = kvPair.second;
        std::string keyName = key.as<std::string>();
        if (keyName == "vertex_program_parameters") {
            parseProgramParameters(value.as<sol::table>());
        }
        std::cout << keyName << std::endl;
    }
    */


    return 0;
}