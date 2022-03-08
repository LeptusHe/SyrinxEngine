#pragma once

#include <vector>
#include <Eigen/Eigen>
#include <implot/implot.h>

class UniformCubicSplineSubdivision {
public:
    bool ExecuteS(std::vector<Eigen::Vector2d>& points, int times = 1)
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
        ImPlot::PlotLine("Subdivision-Uniform Cubic Spline", x, y, pointCount);
    }

private:
    std::vector<Eigen::Vector2d> Subdivision(std::vector<Eigen::Vector2d>& points)
    {
        std::vector<Eigen::Vector2d> result;

        const int pointCount = points.size();
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