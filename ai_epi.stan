// ============================================================================
// 多病原 SEIRS（每个病原体独立的 S-E-I-R + 交叉免疫）+ spline_basis样条 log R0(t)
//   1. 每个病原体各自一套 S-E-I-R，交叉免疫把一部分易感者直接转入 R
//   2. 导出每周隐变量：真实发病人数、易感比例、有效再生数 Rt、R0
//   3. 预测期 R0 同时给出三种外推方式（固定 / 去年同期 / 按当前水平缩放的去年同期）
//   4. 生成模拟轨迹：保留每个后验样本的参数，给 R0(t) 加随机游走扰动后重新模拟
//
// 分块拟合（提速）: 病原体数量和各自的固定参数、先验都由 data 传入，
//   没有交叉免疫联系的病原体可以分开拟合（流感三型 / COVID / RSV / HFMD），后验与一起拟合完全相同。
//   病原体顺序和含义由 R 端的 strains 决定。
// 状态排列: 病原体 k 占第 4k-3 ~ 4k 位，依次为 [S_k, E_k, I_k, R_k]，四者之和为 1
// 可选开关（默认 = 原模型）: rw_order = 2 用二阶随机游走（更平滑）；smooth_bounds = 1 用平滑边界代替 R0 硬截断
// 语法: 新数组语法，需要 rstan >= 2.26
// ============================================================================
functions {
  vector build_b_spline(array[] real t, array[] real ext_knot_pos, int ind, int order) {
    int n = size(t);
    vector[n] b_spline;
    vector[n] w1 = rep_vector(0, n);
    vector[n] w2 = rep_vector(0, n);
    if (order == 1) {
      for (i in 1:n)
        b_spline[i] = (ext_knot_pos[ind] <= t[i]) && (t[i] < ext_knot_pos[ind + 1]);
    } else {
      if (ext_knot_pos[ind] != ext_knot_pos[ind + order - 1])
        w1 = (to_vector(t) - rep_vector(ext_knot_pos[ind], n)) /
             (ext_knot_pos[ind + order - 1] - ext_knot_pos[ind]);
      if (ext_knot_pos[ind + 1] != ext_knot_pos[ind + order])
        w2 = 1 - (to_vector(t) - rep_vector(ext_knot_pos[ind + 1], n)) /
                 (ext_knot_pos[ind + order] - ext_knot_pos[ind + 1]);
      b_spline = w1 .* build_b_spline(t, ext_knot_pos, ind, order - 1) +
                 w2 .* build_b_spline(t, ext_knot_pos, ind + 1, order - 1);
    }
    return b_spline;
  }

  // 由样条系数得到每日 R0 (N x T)。样条基用稀疏形式 (CSR)：每天只有 degree+1 个非零基函数
  // smooth = 0: 截断到 [lo, hi]（原模型）；smooth = 1: 在 log 尺度上用 logistic 平滑压进 (lo, hi)，中间斜率为 1
  matrix R0_daily(array[] row_vector coeff, int T, int nb, vector basis_w, array[] int basis_v,
                  array[] int basis_u, real lo, real hi, int smooth) {
    int N = size(coeff);
    matrix[N, T] R0;
    real a = log(lo);
    real b = log(hi);
    for (k in 1:N) {
      vector[T] lr = csr_matrix_times_vector(T, nb, basis_w, basis_v, basis_u, coeff[k]');
      for (t in 1:T) {
        if (smooth == 1)
          R0[k, t] = exp(a + (b - a) * inv_logit((lr[t] - 0.5 * (a + b)) * 4 / (b - a)));
        else
          R0[k, t] = fmax(lo, fmin(hi, exp(lr[t])));
      }
    }
    return R0;
  }

  // 单日 SEIRS 更新 (dt = 1 天)，带交叉免疫
  // C[j, k]: 感染病原体 j 后获得的对病原体 k 的保护概率
  // 新感染 j 的人中，对 k 易感的比例为 S_k（独立性近似），其中 C[j,k] 的部分直接转入 R_k，
  // 之后和普通免疫一样以 omega_k 衰减回 S_k
  vector seirs_cross_step(vector y, vector beta, vector sig, vector gam, vector om,
                          matrix C, int N) {
    vector[4 * N] y_new;
    vector[N] S;
    vector[N] E;
    vector[N] I;
    vector[N] R;
    vector[N] new_inf;
    vector[N] cross_out;

    for (k in 1:N) {
      S[k] = y[4 * k - 3];
      E[k] = y[4 * k - 2];
      I[k] = y[4 * k - 1];
      R[k] = y[4 * k];
    }
    for (k in 1:N)
      new_inf[k] = fmin(beta[k] * I[k] * S[k], 0.5 * S[k]);

    cross_out = S .* (C' * new_inf);   // cross_out[k] = S_k * sum_j C[j,k] * new_inf[j]

    for (k in 1:N) {
      real ni = new_inf[k];
      real co = cross_out[k];
      real out_k = ni + co;
      if (out_k > 0.9 * S[k]) {        // 流出不超过 S_k 的 90%
        real sc = 0.9 * S[k] / out_k;
        ni *= sc;
        co *= sc;
      }
      y_new[4 * k - 3] = S[k] - ni - co + om[k] * R[k];
      y_new[4 * k - 2] = E[k] + ni - sig[k] * E[k];
      y_new[4 * k - 1] = I[k] + sig[k] * E[k] - gam[k] * I[k];
      y_new[4 * k]     = R[k] + gam[k] * I[k] - om[k] * R[k] + co;
    }

    for (j in 1:(4 * N)) y_new[j] = fmax(1e-10, y_new[j]);
    for (k in 1:N) {
      real tot = sum(y_new[(4 * k - 3):(4 * k)]);
      y_new[(4 * k - 3):(4 * k)] = y_new[(4 * k - 3):(4 * k)] / tot;
    }
    return y_new;
  }

  // 从状态 y0 出发按每日 R0 模拟 n_weeks 周
  // 返回 (n_weeks + 1) x 4N 矩阵:
  //   第 1..n_weeks 行: [每周发病比例 (N) | 周均易感比例 (N) | 周均 Rt (N) | 周均 R0 (N)]
  //   第 n_weeks + 1 行: 模拟结束时的状态向量 (长度正好也是 4N)
  // full = 0 时只算发病比例和期末状态（拟合时用，省掉不进似然的计算）
  matrix simulate_weeks(vector y0, matrix R0_d, vector sig, vector gam, vector om,
                        matrix C, int N, int n_weeks, int full) {
    matrix[n_weeks + 1, 4 * N] out = rep_matrix(0, n_weeks + 1, 4 * N);
    vector[4 * N] y = y0;
    for (w in 1:n_weeks) {
      for (d in 1:7) {
        int t = (w - 1) * 7 + d;
        for (k in 1:N) {
          out[w, k] += sig[k] * y[4 * k - 2];           // E -> I，发病
          if (full == 1) {
            real S = y[4 * k - 3];
            out[w, N + k]     += S / 7;
            out[w, 2 * N + k] += R0_d[k, t] * S / 7;
            out[w, 3 * N + k] += R0_d[k, t] / 7;
          }
        }
        y = seirs_cross_step(y, col(R0_d, t) .* gam, sig, gam, om, C, N);
      }
    }
    out[n_weeks + 1] = y';
    return out;
  }

  // 防止 mu 很大时 neg_binomial_2_rng 溢出报错；gamma 抽到 0 时 poisson_rng 也会报错，一并保护
  int safe_neg_binomial_2_rng(real mu, real phi) {
    real lam = gamma_rng(phi, phi / mu);
    return poisson_rng(fmin(fmax(lam, 1e-12), 1e9));
  }
}

data {
  int<lower=1> T_weeks;                              // 拟合周数
  int<lower=1> N_strains;                            // 本次拟合的病原体数
  array[T_weeks, N_strains] int<lower=-1> cases;     // -1 表示缺失
  int<lower=2> num_knots;
  int<lower=1> spline_degree;

  // 各病原体的固定参数和人群
  vector<lower=0>[N_strains] sigma_E;                // 1 / 潜伏期 (1/天)
  vector<lower=0>[N_strains] gamma_I;                // 1 / 传染期 (1/天)
  vector<lower=0>[N_strains] pop_k;                  // 各病原体对应的人群规模（HFMD 为儿童）

  // 样条先验: 第一个系数 ~ N(mean, sd)，之后随机游走 N(0, tau_rw)（rw_order = 2 时为二阶）
  vector[N_strains] logR0_prior_mean;
  vector<lower=0>[N_strains] logR0_prior_sd;
  real<lower=0> tau_rw_prior_sd;
  int<lower=1, upper=2> rw_order;
  int<lower=0, upper=1> smooth_bounds;
  real<lower=0> R0_min;                              // R0 的下限和上限
  real<lower=R0_min> R0_max;

  // 免疫持续时间先验 (天): lognormal(log(median), sdlog)
  vector<lower=0>[N_strains] immunity_days_prior_median;
  vector<lower=0>[N_strains] immunity_days_prior_sdlog;

  // 观测比例（检出率；HFMD 为住院比例）~ beta(a, b)；过度离散 phi ~ gamma(shape, rate)
  vector<lower=0>[N_strains] obs_prior_a;
  vector<lower=0>[N_strains] obs_prior_b;
  vector<lower=0>[N_strains] phi_prior_shape;
  vector<lower=0>[N_strains] phi_prior_rate;

  // 交叉免疫: cross_map[j, k] = 感染 j 对 k 的保护使用第几个参数，0 = 没有交叉免疫
  int<lower=0> N_cross;
  array[N_strains, N_strains] int<lower=0, upper=N_cross> cross_map;
  vector<lower=0>[N_cross] cross_prior_a;            // cross ~ beta(a, b)
  vector<lower=0>[N_cross] cross_prior_b;

  int<lower=0, upper=52> T_fc_weeks;                 // 向前预测周数

  int<lower=0, upper=1> gen_synthetic;               // 1 = 生成模拟轨迹
  real<lower=0> kappa;                               // 模拟轨迹的 R0 扰动强度 (0 = 不扰动)
}

transformed data {
  // 注意: transformed data 的变量名不要用 X、B 这类短名字 —— rstan 会沿调用栈按名字找同名 R 变量
  //（例如 lapply 内部的 X）当作数据，导致在函数里调用 sampling 时报错
  int T_days = T_weeks * 7;
  int T_fc_days = T_fc_weeks * 7;
  int T_sim = gen_synthetic * T_weeks;
  int num_basis = num_knots + spline_degree - 1;
  int has_last_year = T_days >= 371;                 // 是否有一年以上数据可做季节外推
  array[T_days] real day_frac;
  vector[num_knots] knot_pos;
  vector[2 * spline_degree + num_knots] ext_knot_pos;
  matrix[num_basis, T_days] spline_basis;

  for (k in 1:N_strains)
    if (cross_map[k, k] != 0) reject("cross_map diagonal must be 0");

  for (t in 1:T_days) day_frac[t] = (t - 1.0) / (T_days - 1.0);
  for (k in 1:num_knots) knot_pos[k] = (k - 1.0) / (num_knots - 1.0);

  ext_knot_pos = append_row(append_row(rep_vector(knot_pos[1], spline_degree), knot_pos),
                         rep_vector(knot_pos[num_knots], spline_degree));

  for (ind in 1:num_basis)
    spline_basis[ind, :] = to_row_vector(build_b_spline(day_frac, to_array_1d(ext_knot_pos), ind, spline_degree + 1));
  spline_basis[num_basis, T_days] = 1;

  // spline_basis' 的稀疏形式 (T_days x num_basis)
  vector[num_elements(csr_extract_w(spline_basis'))] basis_w = csr_extract_w(spline_basis');
  array[size(csr_extract_v(spline_basis'))] int basis_v = csr_extract_v(spline_basis');
  array[T_days + 1] int basis_u = csr_extract_u(spline_basis');
}

parameters {
  vector<lower=0, upper=1>[N_strains] susc0;         // 初始时（未在感染中的人里）易感的比例
  vector<upper=0>[N_strains] log_inf0;               // 初始 E+I 占比的对数
  array[N_strains] vector[num_basis] z_spline;       // 非中心化的样条增量
  vector<lower=0>[N_strains] tau_rw;                 // 随机游走步长 (越小越平滑)
  vector<lower=30>[N_strains] immunity_days;         // 平均免疫持续时间 (天)
  vector<lower=0, upper=1>[N_cross] cross;           // 交叉免疫强度
  vector<lower=0, upper=1>[N_strains] obs_rate;      // 检出率（HFMD 为住院比例）
  vector<lower=0>[N_strains] phi;
}

transformed parameters {
  array[N_strains] row_vector[num_basis] log_R0_spline_coeff;
  vector[N_strains] omega = 1.0 ./ immunity_days;    // 免疫衰减率 (1/天)
  matrix[N_strains, N_strains] cross_matrix = rep_matrix(0, N_strains, N_strains);
  vector[4 * N_strains] init_vec;
  vector[N_strains] obs_scale = pop_k .* obs_rate;   // 发病比例 -> 报告病例数
  vector[4 * N_strains] state_end;                   // 拟合期末状态（预测起点）
  matrix[T_weeks, N_strains] mu_cases;               // 每周期望报告病例数

  for (k in 1:N_strains) {
    log_R0_spline_coeff[k, 1] = logR0_prior_mean[k] + logR0_prior_sd[k] * z_spline[k, 1];
    for (b in 2:num_basis) {
      if (rw_order == 2 && b > 2)
        log_R0_spline_coeff[k, b] = 2 * log_R0_spline_coeff[k, b - 1]
                                    - log_R0_spline_coeff[k, b - 2] + tau_rw[k] * z_spline[k, b];
      else
        log_R0_spline_coeff[k, b] = log_R0_spline_coeff[k, b - 1] + tau_rw[k] * z_spline[k, b];
    }
  }

  for (j in 1:N_strains)
    for (k in 1:N_strains)
      if (cross_map[j, k] > 0)
        cross_matrix[j, k] = cross[cross_map[j, k]];

  for (k in 1:N_strains) {
    real inf0 = exp(log_inf0[k]);
    init_vec[4 * k - 3] = susc0[k] * (1 - inf0);
    init_vec[4 * k - 2] = inf0 / 2;
    init_vec[4 * k - 1] = inf0 / 2;
    init_vec[4 * k]     = (1 - susc0[k]) * (1 - inf0);
  }

  {
    matrix[T_weeks + 1, 4 * N_strains] sim =
      simulate_weeks(init_vec,
                     R0_daily(log_R0_spline_coeff, T_days, num_basis, basis_w, basis_v, basis_u, R0_min, R0_max,
                              smooth_bounds),
                     sigma_E, gamma_I, omega, cross_matrix, N_strains, T_weeks, 0);
    for (w in 1:T_weeks)
      for (k in 1:N_strains)
        mu_cases[w, k] = fmax(0.1, sim[w, k] * obs_scale[k]);
    state_end = sim[T_weeks + 1]';
  }
}

model {
  susc0 ~ beta(2, 2);
  log_inf0 ~ normal(log(1e-4), 2);

  for (k in 1:N_strains)
    z_spline[k] ~ std_normal();
  tau_rw ~ normal(0, tau_rw_prior_sd);

  immunity_days ~ lognormal(log(immunity_days_prior_median), immunity_days_prior_sdlog);
  cross ~ beta(cross_prior_a, cross_prior_b);
  obs_rate ~ beta(obs_prior_a, obs_prior_b);
  phi ~ gamma(phi_prior_shape, phi_prior_rate);

  for (w in 1:T_weeks)
    for (k in 1:N_strains)
      if (cases[w, k] >= 0)
        cases[w, k] ~ neg_binomial_2(mu_cases[w, k], phi[k]);
}

generated quantities {
  // ---- 拟合期 ----
  array[T_weeks, N_strains] int pred_cases;          // 后验预测病例
  matrix[T_weeks, N_strains] log_lik;
  vector[N_strains] R0_end;                          // 拟合期最后 7 天的平均 R0
  // ---- 隐变量（每周）----
  matrix[T_weeks, N_strains] inc_week;               // 真实发病人数（未乘检出率）
  matrix[T_weeks, N_strains] S_week;                 // 易感比例（周均值）
  matrix[T_weeks, N_strains] Rt_week;                // 有效再生数 = R0 x S
  matrix[T_weeks, N_strains] R0_week;                // 基本再生数（周均值）

  // ---- 预测期：三种 R0 外推方式 ----
  //   1 = 固定为 R0_end
  //   2 = 去年同一天的 R0
  //   3 = 去年同一天的 R0 x (R0_end / 去年同期 7 天平均 R0)
  //   不足一年数据时 2、3 自动退化为 1
  array[3] matrix[T_fc_weeks, N_strains] mu_fc;
  array[3, T_fc_weeks, N_strains] int pred_fc;
  array[3] matrix[T_fc_weeks, N_strains] Rt_fc;
  array[3] matrix[T_fc_weeks, N_strains] S_fc;

  // ---- 模拟轨迹（gen_synthetic = 1 时才有内容）----
  array[T_sim, N_strains] int y_sim;                 // 模拟报告病例
  matrix[T_sim, N_strains] inc_sim;                  // 模拟真实发病人数
  matrix[T_sim, N_strains] S_sim;                    // 模拟易感比例
  matrix[T_sim, N_strains] Rt_sim;                   // 模拟 Rt

  for (w in 1:T_weeks) {
    for (k in 1:N_strains) {
      pred_cases[w, k] = safe_neg_binomial_2_rng(mu_cases[w, k], phi[k]);
      if (cases[w, k] >= 0)
        log_lik[w, k] = neg_binomial_2_lpmf(cases[w, k] | mu_cases[w, k], phi[k]);
      else
        log_lik[w, k] = 0;
    }
  }

  {
    matrix[N_strains, T_days] R0_d = R0_daily(log_R0_spline_coeff, T_days, num_basis, basis_w, basis_v, basis_u,
                                              R0_min, R0_max, smooth_bounds);
    matrix[T_weeks + 1, 4 * N_strains] sim =
      simulate_weeks(init_vec, R0_d, sigma_E, gamma_I, omega, cross_matrix, N_strains, T_weeks, 1);
    for (w in 1:T_weeks) {
      for (k in 1:N_strains) {
        inc_week[w, k] = sim[w, k] * pop_k[k];
        S_week[w, k]   = sim[w, N_strains + k];
        Rt_week[w, k]  = sim[w, 2 * N_strains + k];
        R0_week[w, k]  = sim[w, 3 * N_strains + k];
      }
    }

    for (k in 1:N_strains)
      R0_end[k] = mean(R0_d[k, (T_days - 6):T_days]);

    for (m in 1:3) {
      matrix[N_strains, T_fc_days] R0_f;
      for (k in 1:N_strains) {
        real ratio = 1;
        if (m == 3 && has_last_year == 1)
          ratio = R0_end[k] / mean(R0_d[k, (T_days - 370):(T_days - 364)]);
        for (d in 1:T_fc_days) {
          real r = R0_end[k];
          if (m > 1 && has_last_year == 1)
            r = R0_d[k, T_days - 364 + d] * ratio;
          R0_f[k, d] = fmax(R0_min, fmin(R0_max, r));
        }
      }
      {
        matrix[T_fc_weeks + 1, 4 * N_strains] fc =
          simulate_weeks(state_end, R0_f, sigma_E, gamma_I, omega, cross_matrix,
                         N_strains, T_fc_weeks, 1);
        for (w in 1:T_fc_weeks) {
          for (k in 1:N_strains) {
            mu_fc[m, w, k]   = fmax(0.1, fc[w, k] * obs_scale[k]);
            pred_fc[m, w, k] = safe_neg_binomial_2_rng(mu_fc[m, w, k], phi[k]);
            S_fc[m, w, k]    = fc[w, N_strains + k];
            Rt_fc[m, w, k]   = fc[w, 2 * N_strains + k];
          }
        }
      }
    }
  }

  if (gen_synthetic == 1) {
    array[N_strains] row_vector[num_basis] coeff_sim;
    for (k in 1:N_strains) {
      real delta = 0;
      for (b in 1:num_basis) {
        delta += kappa * tau_rw[k] * normal_rng(0, 1);
        coeff_sim[k, b] = log_R0_spline_coeff[k, b] + delta;
      }
    }
    {
      matrix[T_weeks + 1, 4 * N_strains] sim =
        simulate_weeks(init_vec,
                       R0_daily(coeff_sim, T_days, num_basis, basis_w, basis_v, basis_u, R0_min, R0_max,
                                smooth_bounds),
                       sigma_E, gamma_I, omega, cross_matrix, N_strains, T_weeks, 1);
      for (w in 1:T_weeks) {
        for (k in 1:N_strains) {
          inc_sim[w, k] = sim[w, k] * pop_k[k];
          S_sim[w, k]   = sim[w, N_strains + k];
          Rt_sim[w, k]  = sim[w, 2 * N_strains + k];
          y_sim[w, k]   = safe_neg_binomial_2_rng(fmax(0.1, sim[w, k] * obs_scale[k]), phi[k]);
        }
      }
    }
  }
}
