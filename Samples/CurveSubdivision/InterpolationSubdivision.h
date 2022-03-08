#pragma once

#include <vector>
#include <Eigen/Eigen>
#include <implot/implot.h>

class InterpolationSubdivision {
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
        ImPlot::PlotLine("Subdivision-Interpolation", x, y, pointCount);
    }

private:
    std::vector<Eigen::Vector2d> Subdivision(std::vector<Eigen::Vector2d>& points, float alpha)
    {
        std::vector<Eigen::Vector2d> result;

        const int pointCount = points.size();
        for (int i = 0; i < pointCount - 1; ++ i) {
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