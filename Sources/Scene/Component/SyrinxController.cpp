#include "Component/SyrinxController.h"
#include <Common/SyrinxAssert.h>

namespace Syrinx {

Controller::Controller()
    : mInput(nullptr)
    , mTransform(nullptr)
    , mEntity(nullptr)
{
    SYRINX_ENSURE(!mInput);
    SYRINX_ENSURE(!mTransform);
    SYRINX_ENSURE(!mEntity);
}


void Controller::setTransform(Transform *transform)
{
    SYRINX_EXPECT(!mTransform);
    SYRINX_EXPECT(transform);
    mTransform = transform;
    SYRINX_ENSURE(mTransform == transform);
}


void Controller::setInput(Input *input)
{
    SYRINX_EXPECT(!mInput);
    SYRINX_EXPECT(input);
    mInput = input;
    SYRINX_ENSURE(mInput == input);
}


void Controller::setEntity(Entity *entity)
{
    SYRINX_EXPECT(!mEntity);
    mEntity = entity;
    SYRINX_EXPECT(mEntity);
    SYRINX_EXPECT(mEntity == entity);
}


const Transform& Controller::getTransform() const
{
    SYRINX_EXPECT(mTransform);
    return *mTransform;
}


const Input& Controller::getInput() const
{
    SYRINX_EXPECT(mInput);
    return *mInput;
}


Transform& Controller::getTransform()
{
    SYRINX_EXPECT(mTransform);
    return *mTransform;
}


Input& Controller::getInput()
{
    SYRINX_EXPECT(mInput);
    return *mInput;
}


Entity& Controller::getEntity() const
{
    SYRINX_EXPECT(mEntity);
    return *mEntity;
}

} // namespace Syrinx
