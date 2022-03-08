include_directories(${sample-required-include-dirs})
add_executable(CurveSubdivision SampleMain.cpp UniformCubicSplineSubdivision.h InterpolationSubdivision.h)
target_link_libraries(CurveSubdivision ${sample-required-libraries})