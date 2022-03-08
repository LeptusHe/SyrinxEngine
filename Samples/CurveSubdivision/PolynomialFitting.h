#pragma once

#include <vector>
#include "NumericalMethod.h"


class PolynomialFitting : public INumericalMethod {
public:
    explicit PolynomialFitting(float lambda) : mLambda(lambda) {}

    void SetLambda(float lambda)
    {
        mLambda = lambda;
    }

    void Solve(const std::vector<Eigen::Vector2d>& points) final
    {
        mPoints = points;

        const auto n = static_cast<int>(points.size());
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
        Eigen::VectorXd y = Eigen::VectorXd::Zero(n);

        for (int i = 0; i < n; ++ i) {
            const auto& p = points[i];
            for (int j = 0; j < n; ++ j) {
                A(i, j) = std::pow(p.x(), j);
            }
            y(i) = p.y();
        }
        mFactors = (A.transpose() * A + mLambda * Eigen::MatrixXd::Identity(n, n)).ldlt().solve(A.transpose() * y);
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
    float mLambda = 1.0f;
    std::vector<Eigen::Vector2d> mPoints;
    Eigen::VectorXd mFactors;
};