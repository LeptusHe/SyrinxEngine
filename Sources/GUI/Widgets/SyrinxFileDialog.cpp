#include "SyrinxFileDialog.h"
#include <imgui/imgui.h>
#include <Common/SyrinxAssert.h>
#include <Exception/SyrinxException.h>

namespace Syrinx {

FileDialog& FileDialog::getInstance()
{
    static FileDialog fileDialog;
    return fileDialog;
}


std::pair<bool, std::string> FileDialog::open(const std::string& title, float width, float height, const std::string& directory)
{
    SYRINX_EXPECT(!title.empty());
    SYRINX_EXPECT(width > 0 && height > 0);
    std::pair<bool, std::string> result;

    setDirectory(directory);
    bool isOpen = true;
    ImGui::Begin(title.c_str(), &isOpen);
    auto fileNames = splitDirectory(mDirectory);
    for (size_t i = 0; i < fileNames.size(); ++ i) {
        if (i != 0) {
            ImGui::SameLine();
        }
        if (ImGui::Button(fileNames[i].c_str())) {
            mDirectory = getSubDirectory(fileNames[i]);
        }
    }

    auto entryList = mFileSystem.getEntryListInDirectory(mDirectory);
    bool isFileSelected = false;

    auto contentRegion = ImGui::GetWindowContentRegionMax();
    auto displayRegionSize = ImVec2(contentRegion.x, contentRegion.y - 80.0f);
    ImGui::BeginChild("##FileDialog_FileList", displayRegionSize);
    for (const auto& entry : entryList) {
        std::string displayedName;

        auto entryPath = mFileSystem.combine(mDirectory, entry);
        if (mFileSystem.directoryExist(entryPath)) {
            displayedName = "[Dir]   ";
        } else {
            isFileSelected = true;
            displayedName = "[File]  ";
        }
        displayedName += entry;

        if (ImGui::Selectable(displayedName.c_str(), entry == mSelectedEntry)) {
            mSelectedEntry = entry;
            if (mFileSystem.directoryExist(entryPath)) {
                mDirectory = entryPath;
            }
        }
    }
    ImGui::EndChild();

    ImGui::BeginChild("##FileDialog_Display");
    ImGui::Text("File Name : ");
    ImGui::SameLine();
    if (isFileSelected) {
        ImGui::Text("%s", mSelectedEntry.c_str());
    } else {
        ImGui::Text("%s", "");
    }
    ImGui::SameLine();
    if (ImGui::Button("Ok")) {
        result = {true, mFileSystem.combine(mDirectory, mSelectedEntry)};
    }
    ImGui::EndChild();

    ImGui::End();
    return result;
}


void FileDialog::setDirectory(const std::string& directory)
{
    if (!directory.empty()) {
        if (!mFileSystem.directoryExist(directory)) {
            SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "[{}] is not a valid directory", directory);
        }
        mDirectory = directory;
    } else {
        mDirectory = (!mDirectory.empty()) ? mDirectory : mFileSystem.getWorkingDirectory();
    }

    if (!mFileSystem.directoryExist(mDirectory)) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "[{}] is not a valid directory", mDirectory);
    }
    SYRINX_ENSURE(!mDirectory.empty());
}


std::vector<std::string> FileDialog::splitDirectory(const std::string& directory) const
{
    std::vector<std::string> fileNames;

    const FileSystem::Path path(directory);
    for (const auto& fileName : path) {
        if ((fileName.empty()) || (fileName == path.root_directory()))
            continue;
        fileNames.push_back(fileName.string());
    }
    return fileNames;
}


std::string FileDialog::getSubDirectory(const std::string& lastFileName)
{
    FileSystem::Path result;

    const FileSystem::Path path(mDirectory);
    for (const auto& fileName : path) {
        if (fileName == path.root_name()) {
            result = fileName.string();
        } else {
            result /= fileName;
        }
        if (fileName == lastFileName)
            break;
    }
    return result.string();
}

} // namespace Syrinx
