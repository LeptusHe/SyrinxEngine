#pragma once

#include <vector>
#include "NumericalMethod.h"


class GaussianInterpolation : public INumericalMethod {
public:
    GaussianInterpolation() = default;

    void SetSigma(float sigma)
    {
        mSigma = sigma;
    }

    void Solve(const std::vector<Eigen::Vector2d>& points) final
    {
        mPoints = points;

        int size = points.size();
        double x = points[size - 1].x() + points[size - 2].x();
        double y = points[size - 1].y() + points[size - 2].y();

        auto internalPoints = points;
        internalPoints.emplace_back(x / 2.0, y / 2.0);

        SolveInternal(internalPoints);
    }

    void SolveInternal(const std::vector<Eigen::Vector2d>& points)
    {
        const int n = points.size();

        Eigen::MatrixXd A = Eigen::MatrixXd::Ones(n, n);
        Eigen::VectorXd y = Eigen::VectorXd::Zero(n);
        for (int i = 0; i < n; ++ i) {
            A(i, 0) = 1;

            double xi = points[i].x();
            for (int j = 1; j < n; ++ j) {
                double xj = points[j - 1].x();
                A(i, j) = GaussianBaseFunction(xi, xj, mSigma);
            }
            y(i) = points[i].y();
        }

        mFactors = A.inverse() * y;
    }

    double F(double x) final
    {
        double sum = 0.0;
        sum += mFactors(0);

        for (int i = 1; i < mFactors.size(); ++ i) {
            const auto& xi = mPoints[i - 1].x();
            sum += mFactors(i) * GaussianBaseFunction(x, xi, mSigma);
        }
        return sum;
    }

private:
    static double GaussianBaseFunction(double x, double xi, double sigma)
    {
        double val = - (x - xi) * (x - xi) / (2 * sigma * sigma);
        return std::exp(val);
    }

private:
    double mSigma = 1.0;
    std::vector<Eigen::Vector2d> mPoints;
    Eigen::VectorXd mFactors;
};