#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <random>

// Griewank函数（n=2）
double griewank(const std::vector<double>&x) {
    double sum = 0.0;
    double product = 1.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += x[i] * x[i];
        product *= std::cos(x[i] / std::sqrt(i + 1)); // i从0开始，公式中i从1开始
    }
    return 1 + sum / 4000.0 - product;
}

// Rastrigin函数（n=2）
double rastrigin(const std::vector<double>& x) {
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += x[i] * x[i] - 10.0 * std::cos(2 * M_PI * x[i]);
    }
    return 20.0 + sum;
}

// 粒子结构体
struct Particle {
    std::vector<double> position;
    std::vector<double> velocity;
    std::vector<double> best_position;
    double best_fitness;
};

// PSO类
class PSO {
public:
    PSO(int num_particles, int dim, double w, double c1, double c2, int max_iter,
        const std::vector<double>& lower_bound, const std::vector<double>& upper_bound)
        : num_particles(num_particles), dim(dim), w(w), c1(c1), c2(c2),
        max_iter(max_iter), lower_bound(lower_bound), upper_bound(upper_bound),
        gen(std::random_device{}()), dist(0.0, 1.0) {
        // 初始化粒子群
        particles.resize(num_particles);
        for (auto& p : particles) {
            p.position.resize(dim);
            p.velocity.resize(dim, 0.0);
            p.best_position.resize(dim);
            for (int i = 0; i < dim; ++i) {
                p.position[i] = lower_bound[i] + (upper_bound[i] - lower_bound[i]) * dist(gen);
                p.velocity[i] = 0.0;
            }
            double fitness = griewank(p.position); // 临时值，后续optimize时会覆盖
            p.best_position = p.position;
            p.best_fitness = fitness;
            if (fitness < global_best_fitness) {
                global_best_fitness = fitness;
                global_best_position = p.position;
            }
        }
    }

    void optimize(const std::function<double(const std::vector<double>&)>& fitness_func) {
        // 重新初始化全局最优
        global_best_fitness = std::numeric_limits<double>::max();
        for (auto& p : particles) {
            double fitness = fitness_func(p.position);
            p.best_fitness = fitness;
            if (fitness < global_best_fitness) {
                global_best_fitness = fitness;
                global_best_position = p.position;
            }
        }

        for (int iter = 0; iter < max_iter; ++iter) {
            for (auto& p : particles) {
                // 更新速度
                for (int i = 0; i < dim; ++i) {
                    double r1 = dist(gen);
                    double r2 = dist(gen);
                    p.velocity[i] = w * p.velocity[i]
                        + c1 * r1 * (p.best_position[i] - p.position[i])
                        + c2 * r2 * (global_best_position[i] - p.position[i]);
                }

                // 更新位置并处理边界
                for (int i = 0; i < dim; ++i) {
                    p.position[i] += p.velocity[i];
                    if (p.position[i] < lower_bound[i]) {
                        p.position[i] = lower_bound[i];
                        p.velocity[i] *= -0.5;
                    }
                    if (p.position[i] > upper_bound[i]) {
                        p.position[i] = upper_bound[i];
                        p.velocity[i] *= -0.5;
                    }
                }

                // 计算适应度
                double current_fitness = fitness_func(p.position);

                // 更新个体最优
                if (current_fitness < p.best_fitness) {
                    p.best_fitness = current_fitness;
                    p.best_position = p.position;
                }

                // 更新全局最优
                if (current_fitness < global_best_fitness) {
                    global_best_fitness = current_fitness;
                    global_best_position = p.position;
                }
            }
        }
    }

    void print_result() const {
        std::cout << "最优解: (";
        for (size_t i = 0; i < global_best_position.size(); ++i) {
            std::cout << global_best_position[i];
            if (i != global_best_position.size() - 1) std::cout << ", ";
        }
        std::cout << ")\n适应度值: " << global_best_fitness << std::endl;
    }

private:
    int num_particles;
    int dim;
    double w;
    double c1;
    double c2;
    int max_iter;
    std::vector<double> lower_bound;
    std::vector<double> upper_bound;
    std::vector<Particle> particles;
    std::vector<double> global_best_position;
    double global_best_fitness;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dist;
};

int main() {
    // 求解Griewank函数
    {
        std::cout << "==== Griewank函数优化 ====\n";
        PSO pso(30, 2, 0.5, 2.0, 2.0, 1000, { -5.0, -5.0 }, { 5.0, 5.0 });
        pso.optimize(griewank);
        pso.print_result();
    }

    // 求解Rastrigin函数
    {
        std::cout << "\n==== Rastrigin函数优化 ====\n";
        PSO pso(50, 2, 0.5, 2.0, 2.0, 1000, { -5.1, -5.12 }, { 5.12, 5.12 });
        pso.optimize(rastrigin);
        pso.print_result();
    }

    return 0;
}
