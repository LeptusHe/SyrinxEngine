#pragma once
#define GLFW_INCLUDE_NONE
#include <string>
#include <memory>
#include <GLFW/glfw3.h>

namespace Syrinx {

class RenderWindow {
public:
    RenderWindow();
	~RenderWindow();
	RenderWindow(const RenderWindow&) = delete;
	RenderWindow& operator=(const RenderWindow&) = delete;

    void setTitle(const std::string& title);
    void setWidth(unsigned width);
    void setHeight(unsigned height);
    void setWindowHandle(GLFWwindow *windowHandle);
    std::string getTitle() const;
    unsigned getWidth() const;
    unsigned getHeight() const;
	const GLFWwindow* getWindowHandle() const;
	GLFWwindow* fetchWindowHandle();
	bool isOpen() const;
	void dispatchEvents() const;
    void swapBuffer() const;

private:
    std::string mTitle;
    unsigned int mWidth;
    unsigned int mHeight;
    GLFWwindow *mWindowHandle;
};

} // namespace Syrinx