#pragma once

#include <vector>
#include "NumericalMethod.h"

class CubicSplineInterpolation : public INumericalMethod {
public:
    CubicSplineInterpolation() = default;

    void Solve(const std::vector<Eigen::Vector2d>& points) final
    {
        mPoints = points;
        mX = std::vector<double>();
        for (const auto& p : points) {
            mX.push_back(p.x());
        }

        const int n = static_cast<int>(points.size()) - 1;
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(4 * n, 4 * n);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(4 * n);

        // interpolation points
        int rowIndex = 0;
        for (int i = 0; i <= n; ++ i) {
            const auto& p = points[i];
            const double x = p.x();
            const double y = p.y();

            const int colIndex = std::min(4 * i, 4 * (n - 1));
            A(i, colIndex + 0) = 1;
            A(i, colIndex + 1) = x;
            A(i, colIndex + 2) = x * x;
            A(i, colIndex + 3) = x * x * x;

            b(i) = y;
            rowIndex += 1;
        }

        // C0
        for (int i = 1; i < n; ++ i) {
            int colIndex = 4 * (i - 1);

            const auto& p = points[i];
            const auto x = p.x();
            for (int k = 0; k < 2; ++ k) {
                colIndex += 4 * k;
                const double scale = -2.0 * static_cast<double>(k) + 1.0;

                A(rowIndex, colIndex + 0) = 1 * scale;
                A(rowIndex, colIndex + 1) = x * scale;
                A(rowIndex, colIndex + 2) = x * x * scale;
                A(rowIndex, colIndex + 3) = x * x * x * scale;
            }
            rowIndex += 1;
        }

        // C1
        for (int i = 1; i < n; ++ i) {
            int colIndex = 4 * (i - 1);

            const auto& p = points[i];
            const auto x = p.x();
            for (int k = 0; k < 2; ++ k) {
                colIndex += 4 * k;
                const double scale = -2.0 * static_cast<double>(k) + 1.0;

                A(rowIndex, colIndex + 0) = 0;
                A(rowIndex, colIndex + 1) = scale;
                A(rowIndex, colIndex + 2) = 2.0 * x * scale;
                A(rowIndex, colIndex + 3) = 3.0 * x * x * scale;
            }
            rowIndex += 1;
        }

        // C2
        for (int i = 1; i < n; ++ i) {
            int colIndex = 4 * (i - 1);

            const auto& p = points[i];
            const auto x = p.x();
            for (int k = 0; k < 2; ++ k) {
                colIndex += 4 * k;
                const double scale = -2.0 * static_cast<double>(k) + 1.0;

                A(rowIndex, colIndex + 0) = 0;
                A(rowIndex, colIndex + 1) = 0;
                A(rowIndex, colIndex + 2) = 2.0 * scale;
                A(rowIndex, colIndex + 3) = 6.0 * x * scale;
            }
            rowIndex += 1;
        }

        // nature c0
        A(rowIndex, 0) = 0;
        A(rowIndex, 1) = 0;
        A(rowIndex, 2) = 2;
        A(rowIndex, 3) = 6 * mX[0];
        rowIndex += 1;

        A(rowIndex, 4 * n - 4) = 0;
        A(rowIndex, 4 * n - 3) = 0;
        A(rowIndex, 4 * n - 2) = 2;
        A(rowIndex, 4 * n - 1) = 6 * mX[mX.size() - 1];
        rowIndex += 1;

        assert(rowIndex == 4 * n);

        mFactors = A.colPivHouseholderQr().solve(b);
    }

    double F(double x) final
    {
        if (x < mX[0])
            return 0.0f;
        if (x > mX[mX.size() - 1])
            return 0.0f;

        int n = mX.size() - 1;
        for (int i = 0; i < n; ++ i) {
            double left = mX[i + 0];
            double right = mX[i + 1];
            if ((left <= x) && (x <= right)) {
                return FInternal(i, x);
            }
        }
        return 0.0f;
    }

    double FInternal(int segmentIndex, double x) const
    {
        int startIndex = 4 * segmentIndex;
        double sum = 0;
        for (int i = 0; i < 4; ++ i) {
            const auto factor = mFactors(startIndex + i);
            sum += factor * std::pow(x, i);
        }
        return sum;
    }

private:
    std::vector<double> mX;
    std::vector<Eigen::Vector2d> mPoints;
    Eigen::VectorXd mFactors;
};
