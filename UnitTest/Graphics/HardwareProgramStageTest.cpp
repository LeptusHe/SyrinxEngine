#include <gmock/gmock.h>
#include <HardwareResource/SyrinxProgramStage.h>

using namespace testing;
using namespace Syrinx;


TEST(ProgramStage, default_state_is_uncreated)
{
    ProgramStage programStage("vertex program");
    ASSERT_THAT(programStage.getState()._value, Eq(HardwareResourceState::Uncreated));
    ASSERT_FALSE(programStage.isCreated());
}


TEST(ProgramStage, default_source_is_empty)
{
    ProgramStage programStage("vertex program");
    ASSERT_TRUE(programStage.getSource().empty());
}


TEST(ProgramStage, default_stage_type_is_undefined_stage)
{
    ProgramStage programStage("vertex program");
    ASSERT_THAT(programStage.getType()._value, Eq(ProgramStageType::UndefinedStage));
}