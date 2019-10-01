#pragma once
#include "SyrinxTransform.h"
#include <Input/SyrinxInput.h>
#include <entityx/entityx.h>

namespace Syrinx {

class Entity;

class Controller {
public:
    using TimeDelta = entityx::TimeDelta;

public:
    Controller();
    virtual ~Controller() = default;

    virtual void update(TimeDelta timeDelta) = 0;
    void setTransform(Transform *transform);
    void setInput(Input *input);
    void setEntity(Entity *entity);
    const Transform& getTransform() const;
    const Input& getInput() const;
    Transform& getTransform();
    Input& getInput();
    Entity& getEntity() const;

private:
    Input *mInput;
    Transform *mTransform;
    Entity *mEntity;
};

} // namespace Syrinx