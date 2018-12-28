#include <gmock/gmock.h>
#include <ResourceManager/SyrinxFileManager.h>

using namespace testing;
using namespace Syrinx;


TEST(FileManager, add_search_path)
{
    auto fileManager = new FileManager();
    ASSERT_ANY_THROW(fileManager->addSearchPath("./notExists"));
}
