#pragma once

#include "SubdivisionMethod.h"

class InterpolationSubdivision : public ISubdivisionMethod {
public:
    bool Execute(std::vector<Eigen::Vector2d>& points, float alpha, int times = 1)
    {
        mPoints = points;

        if (points.size() <= 1)
            return false;

        times = std::max(0, times);
        for (int i = 0; i < times; ++ i) {
            mPoints = Subdivision(mPoints, alpha);
        }
        return true;
    }

private:
    std::string GetMethodName() override
    {
        return "Interpolation";
    }

    static std::vector<Eigen::Vector2d> Subdivision(std::vector<Eigen::Vector2d>& points, float alpha)
    {
        std::vector<Eigen::Vector2d> result;

        const int pointCount = static_cast<int>(points.size());
        for (int i = 0; i < pointCount; ++ i) {
            const auto& p0 = points[GetIndex(i - 1, pointCount)];
            const auto& p1 = points[GetIndex(i + 0, pointCount)];
            const auto& p2 = points[GetIndex(i + 1, pointCount)];
            const auto& p3 = points[GetIndex(i + 2, pointCount)];

            auto p = 0.5 * (p2 + p1) + alpha * (0.5 * (p2 + p1) - 0.5 * (p0 + p2));

            result.emplace_back(p1);
            result.emplace_back(p);
        }
        return result;
    }
};