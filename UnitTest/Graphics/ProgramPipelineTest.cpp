#include <gmock/gmock.h>
#include <HardwareResource/SyrinxProgramPipeline.h>

using namespace testing;
using namespace Syrinx;


TEST(ProgramPipeline, default_constructor)
{
    ProgramPipeline programPipeline("program pipeline");
}