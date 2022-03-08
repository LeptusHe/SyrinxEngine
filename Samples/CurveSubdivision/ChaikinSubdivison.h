#pragma once

#include <Eigen/Eigen>

class ChaikinSubdivison {
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

    const std::vector<Eigen::Vector2d>& GetPoints()
    {
        return mPoints;
    }

    void DrawPlotLine()
    {
        const auto& points = GetPoints();

        const auto pointCount = points.size() + 1;
        auto x = new double[pointCount];
        auto y = new double[pointCount];
        for (int i = 0; i < pointCount; ++i) {
            x[i] = points[i % points.size()].x();
            y[i] = points[i % points.size()].y();
        }
        ImPlot::PlotLine("Chaikin Subdivision", x, y, pointCount);
    }

private:
    std::vector<Eigen::Vector2d> Subdivision(const std::vector<Eigen::Vector2d>& points)
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

    int GetIndex(int index, int pointCount)
    {
        while (index < 0) {
            index += pointCount;
        }
        return index % pointCount;
    }

private:
    std::vector<Eigen::Vector2d> mPoints;
};