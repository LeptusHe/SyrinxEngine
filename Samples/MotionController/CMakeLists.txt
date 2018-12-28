include_directories(${sample-required-include-dirs})
add_executable(MotionControllerSample MotionControllerSample.cpp)
target_link_libraries(MotionControllerSample ${sample-required-libraries})