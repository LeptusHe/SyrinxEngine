// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "SampleRenderer.h"

// our helper library for window handling
#include "GLFWindow.h"
#include <GL/glew.h>
#include "Camera.h"
#include <functional>

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

	bool firstMouse = true;
	float deltaTime = 1.0f / 60.0f;
	const unsigned int SCR_WIDTH = 1200;
	const unsigned int SCR_HEIGHT = 800;
	float lastX = SCR_WIDTH / 2.0f;
	float lastY = SCR_HEIGHT / 2.0f;
	
	void processInput(GLFWwindow* window)
	{
		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
			glfwSetWindowShouldClose(window, true);

		auto& CAMERA = SCamera::getInstance();
		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
			CAMERA.ProcessKeyboard(FORWARD, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
			CAMERA.ProcessKeyboard(BACKWARD, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
			CAMERA.ProcessKeyboard(LEFT, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
			CAMERA.ProcessKeyboard(RIGHT, deltaTime);
	}

	void ProcessMouseMove(GLFWwindow* window, double xpos, double ypos)
	{
		if (firstMouse) {
			lastX = xpos;
			lastY = ypos;
		}
		GLfloat x_offset = xpos - SCR_WIDTH / 2;
		GLfloat y_offset = SCR_HEIGHT / 2 - ypos;
		glfwSetCursorPos(window, SCR_WIDTH / 2, SCR_HEIGHT / 2);
		auto& CAMERA = SCamera::getInstance();
		CAMERA.ProcessMouseMovement(x_offset, y_offset);
	}

	/*
	void mouse_callback(GLFWwindow* window, double xpos, double ypos)
	{
		if (firstMouse)
		{
			lastX = xpos;
			lastY = ypos;
			firstMouse = false;
		}

		float xoffset = xpos - lastX;
		float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

		lastX = xpos;
		lastY = ypos;

		auto& CAMERA = SCamera::getInstance();
		CAMERA.ProcessMouseMovement(xoffset, yoffset);
	}
*/
	
  struct SampleWindow : public GLFCameraWindow
  {
    SampleWindow(const std::string &title,
                 const Model *model,
		         SCamera *sCamera,
                 const Camera &camera,
                 const QuadLight &light,
                 const float worldScale)
      : GLFCameraWindow(title,camera.from,camera.at,camera.up,worldScale)
	  , sample(model,light)
	  , mCamera(sCamera)
		
    {
      sample.setCamera(camera);
	  sample.setCamera(sCamera);
    }
    
    virtual void render() override
    {
		bool resetCount = false;
      if (cameraFrame.modified) {
        sample.setCamera(Camera{ cameraFrame.get_from(),
                                 cameraFrame.get_at(),
                                 cameraFrame.get_up() });
        cameraFrame.modified = false;
		resetCount = true;
      }
      sample.render(resetCount);
    }
    
    virtual void draw() override
    {
	  float* data = pixels.data();
      sample.downloadPixels(data);

      if (fbTexture == 0)
        glGenTextures(1, &fbTexture);
      
      glBindTexture(GL_TEXTURE_2D, fbTexture);
      GLenum texFormat = GL_RGBA32F;
      GLenum texelType = GL_FLOAT;
      glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
                   texelType, pixels.data());

      glDisable(GL_LIGHTING);
      glColor3f(1, 1, 1);

      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      glEnable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, fbTexture);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      
      glDisable(GL_DEPTH_TEST);

      glViewport(0, 0, fbSize.x, fbSize.y);

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

      glBegin(GL_QUADS);
      {
        glTexCoord2f(0.f, 0.f);
        glVertex3f(0.f, 0.f, 0.f);
      
        glTexCoord2f(0.f, 1.f);
        glVertex3f(0.f, (float)fbSize.y, 0.f);
      
        glTexCoord2f(1.f, 1.f);
        glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);
      
        glTexCoord2f(1.f, 0.f);
        glVertex3f((float)fbSize.x, 0.f, 0.f);
      }
      glEnd();
    }
    
    virtual void resize(const vec2i &newSize) 
    {
      fbSize = newSize;
      sample.resize(newSize);
      pixels.resize(newSize.x * newSize.y * 4.0);
    }

    vec2i                 fbSize;
    GLuint                fbTexture {0};
    SampleRenderer        sample;
    std::vector<float> pixels;
	SCamera *mCamera;
  };
  
  
  /*! main entry point to this example - initially optix, print hello
    world, then exit */
  extern "C" int main(int ac, char **av)
  {
    try {
      Model *model = loadOBJ(
#ifdef _WIN32
      // on windows, visual studio creates _two_ levels of build dir
      // (x86/Release)
      "Medias/Models/sponzaOrig/sponza.obj"
#else
      // on linux, common practice is to have ONE level of build dir
      // (say, <project>/build/)...
      "../models/sponza.obj"
#endif
                             );
      Camera camera = { /*from*/vec3f(-1293.07f, 154.681f, -0.7304f),
                        /* at */model->bounds.center()-vec3f(0,400,0),
                        /* up */vec3f(0.f,1.f,0.f) };

      // some simple, hard-coded light ... obviously, only works for sponza
      const float light_size = 200.f;
      QuadLight light = { /* origin */ vec3f(-1000-light_size,800,-light_size),
                          /* edge 1 */ vec3f(2.f*light_size,0,0),
                          /* edge 2 */ vec3f(0,0,2.f*light_size),
                          /* power */  vec3f(3000000.f) };
                      
      // something approximating the scale of the world, so the
      // camera knows how much to move for any given user interaction:
      const float worldScale = length(model->bounds.span());

		auto& CAMERA = SCamera::getInstance();
      SampleWindow *window = new SampleWindow("Optix 7 Course Example",
                                              model, &CAMERA, camera,light,worldScale);

	  auto handle = window->handle;
	  window->mInputProcessor = processInput;
      //glfwSetKeyCallback(handle, processInput);
      glfwSetCursorPosCallback(handle, ProcessMouseMove);

      window->run();
      
    } catch (std::runtime_error& e) {
      std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
                << GDT_TERMINAL_DEFAULT << std::endl;
	  std::cout << "Did you forget to copy sponza.obj and sponza.mtl into your optix7course/models directory?" << std::endl;
	  exit(1);
    }
    return 0;
  }
  
} // ::osc
