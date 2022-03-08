#pragma once

#include "SubdivisionMethod.h"

class ChaikinSubdivision : public ISubdivisionMethod {
public:
    bool Execute(const std::vector<Eigen::Vector2d>& points, int times = 1)
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
        return "Chaikin";
    }

    static std::vector<Eigen::Vector2d> Subdivision(const std::vector<Eigen::Vector2d>& points)
    {
        std::vector<Eigen::Vector2d> result;

        const int pointCount = static_cast<int>(points.size());
        for (int i = 0; i < pointCount; ++ i) {
            const auto& lhs = points[GetIndex(i - 1, pointCount)];
            const auto& rhs = points[GetIndex(i + 0, pointCount)];
            auto mid = (lhs + rhs) / 2;

            const auto newLhs = (lhs + mid) / 2;
            const auto newRhs = (rhs + mid) / 2;

            result.emplace_back(newLhs);
            result.emplace_back(newRhs);
        }
        return result;
    }
};