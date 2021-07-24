#pragma once

#include <vector>
#include "NumericalMethod.h"


class PolynomialInterpolation : public INumericalMethod {
public:
    PolynomialInterpolation() = default;

    void Solve(const std::vector<Eigen::Vector2d>& points) final
    {
        mPoints = points;

        const auto n = static_cast<int>(points.size());
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
        Eigen::VectorXd y = Eigen::VectorXd::Zero(n);

        for (int i = 0; i < n; ++ i) {
            const auto& p = points[i];
            for (int j = 0; j < n; ++ j) {
                A(i, j) = std::pow(p.x(), static_cast<float>(j));
            }
            y(i) = p.y();
        }
        mFactors = A.inverse() * y;
    }

    double F(double x) final
    {
        double result = 0;
        for (int i = 0; i < mFactors.size(); ++ i) {
            result += mFactors(i) * std::pow(x, i);
        }
        return result;
    }

private:
    std::vector<Eigen::Vector2d> mPoints;
    Eigen::VectorXd mFactors;
};;