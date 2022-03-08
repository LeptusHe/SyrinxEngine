#pragma once

#include "SubdivisionMethod.h"

class UniformCubicSplineSubdivision : public ISubdivisionMethod {
public:
    bool Execute(std::vector<Eigen::Vector2d>& points, int times = 1)
    {
        mPoints = points;

        if (points.size() <= 1)
            return false;

        times = std::max(0, times);
        for (int i = 0; i < times; ++ i) {
            mPoints = Subdivision(mPoints);
        }
        return true;
    }

private:
    std::string GetMethodName() override
    {
        return "Uniform Cubic Spline";
    }

    static std::vector<Eigen::Vector2d> Subdivision(std::vector<Eigen::Vector2d>& points)
    {
        std::vector<Eigen::Vector2d> result;

        const int pointCount = static_cast<int>(points.size());
        for (int i = 0; i < pointCount; ++ i) {
            auto lhs = 1.0 / 8.0 * points[GetIndex(i - 1, pointCount)] +
                                          3.0 / 4.0 * points[GetIndex(i, pointCount)] +
                                          1.0 / 8.0 * points[GetIndex(i + 1, pointCount)];
            result.emplace_back(lhs);

            auto rhs = 0.5 * points[GetIndex(i, pointCount)] + 0.5 * points[GetIndex(i + 1, pointCount)];
            result.emplace_back(rhs);
        }
        return result;
    }
};