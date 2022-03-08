#pragma once
#include <cassert>
#include "NumericalMethod.h"

class BezierSegmentCurve {
public:
    Eigen::Vector2d p0, p1;
    Eigen::Vector2d c0, c1;

public:
    double F(double x) const
    {
        assert(x >= p0.x() && x <= p1.x());
        double t = (x - p0.x()) / (p1.x() - p0.x());

        const double tmp = 1 - t;

        double f0 = tmp * tmp * tmp;
        double f1 = 3 * t * tmp * tmp;
        double f2 = 3 * t * t * tmp;
        double f3 = t * t * t;
        return f0 * p0.y() + f1 * c0.y() + f2 * c1.y() + f3 * p1.y();
    }
};

class BezierFitting : public INumericalMethod {
public:
    void Solve(const std::vector<Eigen::Vector2d>& points) final
    {
        int pointCnt = static_cast<int>(points.size());
        int segmentCnt = pointCnt - 1;
        mSegmentCurveList.resize(segmentCnt);

        for (int i = 1; i < segmentCnt - 1; ++ i) {
            BezierSegmentCurve& curve = mSegmentCurveList[i];
            curve.p0 = points[i + 0];
            curve.p1 = points[i + 1];
            curve.c0 = GetRightControlPoint(points[i - 1], points[i], points[i + 1]);
            curve.c1 = GetLeftControlPoint(points[i], points[i+1], points[i + 2]);
        }

        BezierSegmentCurve& startSegment = mSegmentCurveList[0];
        startSegment.p0 = points[0];
        startSegment.p1 = points[1];
        startSegment.c0 = points[0] + Eigen::Vector2d(0, 0.0);
        if (pointCnt >= 3) {
            startSegment.c1 = GetLeftControlPoint(points[0], points[1], points[2]);
        }

        BezierSegmentCurve& endSegment = mSegmentCurveList[segmentCnt - 1];
        endSegment.p0 = points[pointCnt - 2];
        endSegment.p1 = points[pointCnt - 1];
        if (pointCnt >= 3) {
            endSegment.c0 = GetRightControlPoint(points[pointCnt - 3], points[pointCnt - 2], points[pointCnt - 1]);
        }
        endSegment.c1 = points[pointCnt - 1] + Eigen::Vector2d(0, 0.0);
    }

    double F(double x) final
    {
        double result = 0;
        for (int i = 0; i < mSegmentCurveList.size(); ++ i) {
            const auto& segmentCurve = mSegmentCurveList[i];
            double lhs = segmentCurve.p0.x();
            double rhs = segmentCurve.p1.x();

            if (lhs <= x && x <= rhs) {
                return segmentCurve.F(x);
            }
        }
        return result;
    }

private:
    bool IsValid(int pointCnt)
    {
        return pointCnt % 3 == 1;
    }

    Eigen::Vector2d GetLeftControlPoint(const Eigen::Vector2d& p0, const Eigen::Vector2d& p1, const Eigen::Vector2d& p2) const
    {
        return GetControlPoint(p0, p1, p2, -1.0 / 6.0);
    }

    Eigen::Vector2d GetRightControlPoint(const Eigen::Vector2d& p0, const Eigen::Vector2d& p1, const Eigen::Vector2d& p2) const
    {
        return GetControlPoint(p0, p1, p2, 1.0 / 6.0);
    }

    Eigen::Vector2d GetControlPoint(const Eigen::Vector2d& p0, const Eigen::Vector2d& p1, const Eigen::Vector2d& p2, double scale) const
    {
        Eigen::Vector2d dir = p2 - p0;
        return p1 + scale * dir;
    }

private:
    std::vector<BezierSegmentCurve> mSegmentCurveList;
};