#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <RenderSystem/SyrinxCamera.h>

using namespace Syrinx;

CCamera::CCamera(glm::vec3 position, glm::vec3 front, glm::vec3 up)
        : mMovementSpeed(SPEED), mMouseSensitivity(SENSITIVITY), mZoom(ZOOM)
{
    mPosition = position;
    mFront = glm::normalize(front);
    mUp = up;
    mRight = glm::normalize(glm::cross(mUp, -mFront));
    mPitch = asin(mFront.y);
    mYaw = asin(mFront.z / cos(mPitch));
}


void CCamera:: ProcessKeyboard(Camera_Movement direction, float deltaTime)
{
    float velocity = mMovementSpeed * deltaTime;
    if (direction == FORWARD)
        mPosition += mFront * velocity;
    if (direction == BACKWARD)
        mPosition -= mFront * velocity;
    if (direction == LEFT)
        mPosition -= mRight * velocity;
    if (direction == RIGHT)
        mPosition += mRight * velocity;
}


void CCamera::ProcessMouseMovement(float xoffset, float yoffset, bool constrainPitch)
{
    xoffset *= mMouseSensitivity;
    yoffset *= mMouseSensitivity;

    mYaw   += glm::radians(xoffset);
    mPitch += glm::radians(yoffset);

    if (constrainPitch)
    {
        if (mPitch > glm::radians(89.0f))
            mPitch = glm::radians(89.0f);
        if (mPitch < glm::radians(-89.0f))
            mPitch = glm::radians(-89.0f);
    }
    updateCameraVectors();
}


void CCamera::updateCameraVectors()
{
    glm::vec3 front;
    front.x = cos(mYaw) * cos(mPitch);
    front.y = sin(mPitch);
    front.z = sin(mYaw) * cos(mPitch);
    mFront = glm::normalize(front);
    mRight = glm::normalize(glm::cross(mUp, -mFront));
}


const glm::vec3 &CCamera::getPosition() const {
    return mPosition;
}


const glm::vec3 &CCamera::getFront() const {
    return mFront;
}


const glm::vec3 &CCamera::getUp() const {
    return mUp;
}