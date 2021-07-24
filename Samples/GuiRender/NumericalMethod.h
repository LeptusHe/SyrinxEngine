#pragma once

#include <Eigen/Eigen>

class INumericalMethod {
public:
    virtual void Solve(const std::vector<Eigen::Vector2d>& points) = 0;
    virtual double F(double x) = 0;
};