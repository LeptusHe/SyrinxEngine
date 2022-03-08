#pragma once

#include <Eigen/Eigen>
#include <implot/implot.h>

class ISubdivisionMethod {
public:
    void DrawPlotLine()
    {
        std::string label = std::string("subdivision-") + GetMethodName();

        const auto& points = GetPoints();

        const int pointCount = static_cast<int>(points.size()) + 1;
        auto x = new double[pointCount];
        auto y = new double[pointCount];
        for (int i = 0; i < pointCount; ++i) {
            x[i] = points[i % points.size()].x();
            y[i] = points[i % points.size()].y();
        }
        ImPlot::PlotLine(label.c_str(), x, y, pointCount);

        delete[] x;
        delete[] y;
    }

    void DrawPoints()
    {
        auto label = std::string("points-") + GetMethodName();

        const auto& points = GetPoints();
        const int pointCount = static_cast<int>(points.size());

        auto x = new double[pointCount];
        auto y = new double[pointCount];
        for (int i = 0; i < pointCount; ++i) {
            x[i] = points[i % points.size()].x();
            y[i] = points[i % points.size()].y();
        }
        ImPlot::PlotScatter(label.c_str(), x, y, pointCount);

        delete[] x;
        delete[] y;
    }

protected:
    virtual std::string GetMethodName() = 0;

    static int GetIndex(int index, int pointCount)
    {
        while (index < 0) {
            index += pointCount;
        }
        return index % pointCount;
    }

    const std::vector<Eigen::Vector2d>& GetPoints()
    {
        return mPoints;
    }

protected:
    std::vector<Eigen::Vector2d> mPoints;
};
