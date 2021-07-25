#pragma once
#include <exception>
#include <memory>
#include "PolynomialFitting.h"

class IParametrizationMethod {
public:
    virtual std::vector<float> Parametric(const std::vector<Eigen::Vector2d>& points) = 0;
};


class UniformParametrizationMethod : public IParametrizationMethod {
public:
    std::vector<float> Parametric(const std::vector<Eigen::Vector2d>& points) override
    {
        std::vector<float> result;
        int count = static_cast<int>(points.size());
        for (int i = 0; i < count; ++ i) {
            float t = static_cast<float>(i) / static_cast<float>(count - 1);
            result.push_back(t);
        }
        return result;
    }
};


class ChordalParametrization : public IParametrizationMethod {
public:
    std::vector<float> Parametric(const std::vector<Eigen::Vector2d>& points) override
    {
        if (points.size() <= 1) {
            return {};
        }

        float chordalSum = 0.0f;
        std::vector<float> result = {0};
        for (int i = 1; i < points.size(); ++ i) {
            float chordal = CalculateLength(points[i - 1], points[i]);
            result.push_back(chordal);
            chordalSum += chordal;
        }

        float chordalCnt = 0.0f;
        for (float& t : result) {
            chordalCnt += t;
            t = chordalCnt / chordalSum;
        }
        return result;
    }

private:
    float CalculateLength(const Eigen::Vector2d& lhs, const Eigen::Vector2d& rhs) const
    {
        const auto result = rhs - lhs;
        return result.norm();
    }
};


enum class ParametrizationMethod {
    Uniform,
    Chordal,
    Count
};



class ParametricCurve {
public:
    explicit ParametricCurve(ParametrizationMethod method, float lambda) : m_xFitting(lambda), m_yFitting(lambda)
    {
        m_parametrizationMethod = GetParametrizationMethod(method);
    }

    void SetLambda(float lambda)
    {
        m_xFitting = PolynomialFitting(lambda);
        m_yFitting = PolynomialFitting(lambda);
    }

    void Solve(const std::vector<Eigen::Vector2d>& points)
    {
        m_paramsT = m_parametrizationMethod->Parametric(points);

        std::vector<Eigen::Vector2d> xList;
        std::vector<Eigen::Vector2d> yList;
        for (int i = 0; i < static_cast<int>(points.size()); ++ i) {
            float t = m_paramsT[i];
            const auto& p = points[i];

            xList.emplace_back(t, p.x());
            yList.emplace_back(t, p.y());
        }

        m_xFitting.Solve(xList);
        m_yFitting.Solve(yList);
    }

    Eigen::Vector2d F(double t)
    {
        double x = m_xFitting.F(t);
        double y = m_yFitting.F(t);
        return Eigen::Vector2d(x, y);
    }

private:
    static std::unique_ptr<IParametrizationMethod> GetParametrizationMethod(ParametrizationMethod method)
    {
        switch (method) {
            case ParametrizationMethod::Uniform: return std::make_unique<UniformParametrizationMethod>();
            case ParametrizationMethod::Chordal: return std::make_unique<ChordalParametrization>();
            default: {
                throw std::exception("invalid parametrization method");
            }
        }
    }

private:
    std::unique_ptr<IParametrizationMethod> m_parametrizationMethod;
    std::vector<float> m_paramsT;
    PolynomialFitting m_xFitting;
    PolynomialFitting m_yFitting;
};