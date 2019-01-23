#include <iostream>
#include <cxxopts.hpp>
#include <assimp/postprocess.h>
#include <FileSystem/SyrinxFileSystem.h>
#include <Logging/SyrinxLogManager.h>
#include "SyrinxModelExporter.h"
#include "SyrinxExporterOptions.h"


int main(int argc, char *argv[])
{
    const std::string usage = "Usage: SyrinxModelExporter [options]\n"
                              "options:\n"
                              "    -f,--file=[model file name]: model file\n"
                              "    -o,--output=[output dir]: output directory\n"
                              "    -c,--color : generate color in material\n"
                              "    -s,--shader=[shader-file-name]: add shader-file-name into material\n";
    std::cout << usage << std::endl;


    auto logger = std::make_unique<Syrinx::LogManager>();

    cxxopts::Options options(argv[0], "model exporter for syrinx engine");
    options.add_options()
            ("f,file", "model file", cxxopts::value<std::string>())
            ("o,output", "output dir", cxxopts::value<std::string>())
            ("c,color", "parse material color")
            ("s,shader", "add shader file into material", cxxopts::value<std::string>());

    std::string modelFilePath;
    try {
        auto result = options.parse(argc, argv);

        if (result.count("f") == 0) {
            SYRINX_FAULT("fail to get model file from command line");
            return 0;
        }

        Syrinx::FileSystem fileSystem;
        modelFilePath = fileSystem.weaklyCanonical(result["f"].as<std::string>());
        std::string outputDirectory = fileSystem.getParentPath(modelFilePath);
        if (result.count("o") != 0) {
            outputDirectory = fileSystem.weaklyCanonical(result["o"].as<std::string>());
        }

        Syrinx::Tool::ExporterOptions exporterOptions;
        if (result.count("c") != 0) {
            exporterOptions.exportMaterialColor = true;
        }

        if (result.count("s") != 0) {
            exporterOptions.shaderFileName = result["s"].as<std::string>();
        }

        auto fileManager = std::make_unique<Syrinx::FileManager>();
        Syrinx::Tool::ModelExporter modelExporter(fileManager.get());
        modelExporter.exportModel(modelFilePath, outputDirectory, exporterOptions);
    } catch (cxxopts::OptionException& e) {
        SYRINX_FAULT_FMT("fail to parse options: [{}]", e.what());
    } catch (std::exception& e) {
        SYRINX_FAULT_FMT("fail to export model [{}] because [{}]", modelFilePath, e.what());
    }

    return 0;
}