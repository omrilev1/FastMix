import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tqdm import tqdm 
from utils_fast_mixing import * 
import warnings
import matplotlib.ticker as mticker
from time import perf_counter_ns

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")  # suppress all warnings from all modules

def fill_standard_normal(rng, out):
    try:
        rng.standard_normal(out=out)
    except TypeError:
        out[...] = rng.standard_normal(size=out.shape)
    return out

def dot_into(lhs, rhs, out):
    try:
        np.dot(lhs, rhs, out=out)
    except TypeError:
        out[...] = lhs @ rhs
    return out

# Direct Laplace privatization of the sketch regularization parameter.
def direct_private_sigma_tilde(gamma, m_hat, delta_hat, lambda_hat, eta, tau, rng):
    z1 = rng.laplace()
    m_tilde = max(1.0, m_hat + eta * delta_hat * (tau - z1))

    z2 = rng.laplace()
    lambda_tilde = max(0.0, lambda_hat - eta * (1.0 + 2.0*m_tilde) * (tau - z2))
    sigma_sq = gamma * (1.0 + 2.0 * m_tilde) - lambda_tilde

    return np.sqrt(max(0.0, sigma_sq))

def largest_power_of_two_leq(value):
    value = int(value)
    if value < 1:
        raise ValueError("value must be positive")
    return 1 << (value.bit_length() - 1)

def normalize_fast_ihm_inner_percentages(values):
    if isinstance(values, np.ndarray):
        normalized_values = values.tolist()
    elif isinstance(values, (list, tuple)):
        normalized_values = list(values)
    else:
        normalized_values = [values]

    if len(normalized_values) == 0:
        raise ValueError("percentage_k_FastIHM_inner must contain at least one value")

    return normalized_values

def get_fast_ihm_inner_plot_metadata(percentage_value):
    if percentage_value is None:
        return r"$k_{\mathrm{inner}} = k_{\mathrm{outer}}$", "outer"

    percentage_value = float(percentage_value)
    return (
        r"$\frac{k_{\mathrm{inner}}}{\max\{d,\log(\frac{4T}{\varrho})\}} = "
        + f"{percentage_value:.1f}$",
        f"{percentage_value:.1f}",
    )

def empirical_excess_risk_curve(empirical_risk_values, empirical_risk_std, baseline_train_mse_mean):
    if empirical_risk_values is None or empirical_risk_std is None or baseline_train_mse_mean is None:
        return None, None

    excess_values = np.asarray(empirical_risk_values, dtype=float) - float(baseline_train_mse_mean)
    excess_std = np.asarray(empirical_risk_std, dtype=float)
    return excess_values, excess_std

def plot_positive_excess_risk_curve(ax, x_values, excess_values, excess_std, *, color, marker, label, linestyle='-'):
    excess_values = np.asarray(excess_values, dtype=float)
    excess_std = np.asarray(excess_std, dtype=float)

    positive_excess_values = np.where(excess_values > 0.0, excess_values, np.nan)
    lower_band = excess_values - excess_std
    upper_band = excess_values + excess_std
    positive_band = lower_band > 0.0

    if not np.any(np.isfinite(positive_excess_values)):
        return False

    ax.plot(
        x_values,
        positive_excess_values,
        linestyle=linestyle,
        color=color,
        marker=marker,
        markersize=30,
        label=label,
    )

    if np.any(positive_band):
        ax.fill_between(
            x_values,
            np.where(positive_band, lower_band, np.nan),
            np.where(positive_band, upper_band, np.nan),
            color=color,
            alpha=0.2,
        )

    return True

def collect_positive_log_y_bounds(y_bounds, values, std=None):
    if values is None:
        return

    values = np.asarray(values, dtype=float)
    finite_positive_values = values[np.isfinite(values) & (values > 0.0)]
    if finite_positive_values.size > 0:
        y_bounds.append((
            float(np.min(finite_positive_values)),
            float(np.max(finite_positive_values)),
        ))

    if std is None:
        return

    std = np.asarray(std, dtype=float)
    lower_band = values - std
    upper_band = values + std
    positive_band = (
        np.isfinite(lower_band)
        & np.isfinite(upper_band)
        & (lower_band > 0.0)
        & (upper_band > 0.0)
    )
    if np.any(positive_band):
        y_bounds.append((
            float(np.min(lower_band[positive_band])),
            float(np.max(upper_band[positive_band])),
        ))

def set_log_y_limits_from_bounds(ax, y_bounds, min_log_padding_decades=0.08):
    if len(y_bounds) == 0:
        return

    ymin = min(bound[0] for bound in y_bounds)
    ymax = max(bound[1] for bound in y_bounds)
    if (
        not np.isfinite(ymin)
        or not np.isfinite(ymax)
        or ymin <= 0.0
        or ymax <= 0.0
    ):
        return

    if np.isclose(ymin, ymax):
        ymin /= 10.0 ** min_log_padding_decades
        ymax *= 10.0 ** min_log_padding_decades
    else:
        log_ymin = np.log10(ymin)
        log_ymax = np.log10(ymax)
        log_span = max(log_ymax - log_ymin, 1e-6)
        log_padding = max(0.05 * log_span, min_log_padding_decades)
        ymin = 10.0 ** (log_ymin - log_padding)
        ymax = 10.0 ** (log_ymax + log_padding)

    ax.set_ylim(ymin, ymax)

class ScalarFormatterFixed(mticker.ScalarFormatter):
    def __init__(self, fmt="%.2f", **kwargs):
        super().__init__(**kwargs)
        self.fmt = fmt

    def _set_format(self, *args, **kwargs):
        self.format = self.fmt

def fixed_scientific_y_formatter(fmt="%.2f"):
    formatter = ScalarFormatterFixed(fmt=fmt, useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    return formatter

def set_log_y_ticks(ax, min_ticks=3):
    ymin, ymax = ax.get_ylim()
    if not np.all(np.isfinite([ymin, ymax])) or ymin <= 0.0 or ymax <= 0.0 or ymin >= ymax:
        return

    default_locator = mticker.LogLocator(base=10.0)
    default_ticks = np.asarray(default_locator.tick_values(ymin, ymax), dtype=float)
    default_ticks = default_ticks[(default_ticks >= ymin) & (default_ticks <= ymax)]

    if default_ticks.size >= min_ticks:
        ax.yaxis.set_major_locator(default_locator)
        ax.yaxis.set_major_formatter(fixed_scientific_y_formatter())
    else:
        custom_ticks = np.geomspace(ymin, ymax, min_ticks)
        ax.yaxis.set_major_locator(mticker.FixedLocator(custom_ticks))
        ax.yaxis.set_major_formatter(fixed_scientific_y_formatter())

    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

def fast_ihs_line_variant_file_token(add_fast_ihs_baseline_horizontal_line):
    return "fastihsline_on" if add_fast_ihs_baseline_horizontal_line else "fastihsline_off"

def file_token_from_float(value):
    return f"{float(value):.3g}".replace("-", "m").replace(".", "p")

def format_speedup_text(reference_runtime, candidate_runtime, label):
    if reference_runtime is None or candidate_runtime is None:
        return None

    reference_runtime = float(reference_runtime)
    candidate_runtime = float(candidate_runtime)
    if (
        not np.isfinite(reference_runtime)
        or not np.isfinite(candidate_runtime)
        or reference_runtime <= 0.0
        or candidate_runtime <= 0.0
    ):
        return None

    speedup = reference_runtime / candidate_runtime
    if speedup >= 1.0:
        return f"{speedup:.2f}x faster"

    slowdown = candidate_runtime / reference_runtime
    return f"{slowdown:.2f}x slower"

def save_speedup_summary(
    plot_records,
    *,
    dataset_type,
    percentage_k_FastIHS_baseline,
    iters_IHM,
    eigval_noise,
):
    if len(plot_records) == 0:
        return

    inner_tokens = [
        str(plot_record["fast_ihm_inner_file_token"]).replace(".", "p")
        for plot_record in plot_records
    ]
    inner_summary_token = compact_filename_token(inner_tokens, prefix="kf")
    baseline_token = file_token_from_float(percentage_k_FastIHS_baseline)
    speedup_path = os.path.join(
        "timings",
        f"spd_{dataset_type}_kb{baseline_token}_{inner_summary_token}_t{iters_IHM}_{eigval_noise}.txt",
    )

    with open(filesystem_safe_output_path(speedup_path), "w") as f:
        f.write(f"Dataset: {dataset_type}\n")
        f.write(f"percentage_k_FastIHS_baseline: {percentage_k_FastIHS_baseline:.1f}\n")
        f.write(f"IHM iterations: {iters_IHM}\n")
        f.write(f"Eigenvalue noise: {eigval_noise}\n")

        baseline_record = plot_records[0]
        baseline_runtime_by_method = baseline_record.get("overall_runtime_by_method", {})
        ihm_runtime = baseline_runtime_by_method.get("HessianMix")
        fast_ihs_runtime = baseline_runtime_by_method.get("FastIHSBaseline")
        if ihm_runtime is not None:
            f.write(f"IHM overall average runtime: {float(ihm_runtime):.10f}\n")
        if fast_ihs_runtime is not None:
            f.write(f"Fast IHS baseline overall average runtime: {float(fast_ihs_runtime):.10f}\n")
            speedup_text = format_speedup_text(ihm_runtime, fast_ihs_runtime, "Fast IHS baseline")
            if speedup_text is not None:
                f.write(f"Fast IHS baseline vs IHM: {speedup_text}\n")

        f.write("Fast IHM speedups vs IHM:\n")
        for plot_record in plot_records:
            runtime_by_method = plot_record.get("overall_runtime_by_method", {})
            fast_ihm_runtime = runtime_by_method.get("FastHessianMix")
            ihm_runtime_record = runtime_by_method.get("HessianMix")
            speedup_text = format_speedup_text(ihm_runtime_record, fast_ihm_runtime, "Fast IHM")
            if speedup_text is None:
                continue

            inner_percentage = plot_record.get("fast_ihm_inner_percentage")
            if inner_percentage is None:
                inner_label = "outer"
            else:
                inner_label = f"{float(inner_percentage):.1f}"

            f.write(
                f"  inner_k={inner_label}: Fast IHM runtime = {float(fast_ihm_runtime):.10f}, "
                f"IHM runtime = {float(ihm_runtime_record):.10f}, {speedup_text}\n"
            )

def _save_empirical_excess_risk_plot_variant(
    plot_records,
    *,
    dataset_type,
    n,
    d,
    percentage_k_Mix,
    percentage_k_IHM,
    iters_IHM,
    eigval_noise,
    epsilon_values,
    add_legend_to_plot,
    combine_fast_ihm_inner_curves,
    add_fast_ihs_baseline_horizontal_line,
):
    if len(plot_records) == 0:
        return

    label_suffix = r"$\frac{k}{\max\{d,\log(\frac{1}{\varrho})\}} = " + f"{percentage_k_Mix:.1f}$"
    label_suffix_iter = r"$\frac{k}{\max\{d,\log(\frac{4T}{\varrho})\}} = " + f"{percentage_k_IHM:.1f}$"

    colors = {
        'linmix': 'orangered',
        'hessian_mixing': 'green',
        'fast_hessian_mixing': 'blue',
        'gaussian_ihs_baseline': 'firebrick',
        'fast_ihs_baseline': 'teal',
        'gaussian_sketch_solve_baseline': 'darkslategray',
        'fast_linmix': 'black',
        'adassp': 'goldenrod',
    }

    markers = {
        'linmix': 'o',
        'hessian_mixing': 's',
        'fast_hessian_mixing': 'P',
        'fast_linmix': 'D',
        'adassp': '^',
    }

    fast_ihm_colors = ['navy', 'royalblue', 'deepskyblue', 'slateblue', 'teal', 'cornflowerblue']
    fast_ihm_markers = ['P', 'X', '^', 'v', '<', '>', 'D', '*']

    eps = np.asarray(epsilon_values, dtype=float)
    base_record = plot_records[0]
    fast_ihm_records = plot_records if combine_fast_ihm_inner_curves else plot_records[:1]
    use_multi_fast_ihm_style = combine_fast_ihm_inner_curves and len(fast_ihm_records) > 1

    excess_fig = plt.figure(figsize=(10, 7))
    excess_ax = plt.gca()
    plotted_excess = False
    excess_y_bounds = []

    linmix_train_excess, linmix_train_excess_std = empirical_excess_risk_curve(
        base_record["train_mse_for_sigmas"],
        base_record["train_mse_for_sigmas_std"],
        base_record["baseline_train_mse_mean"],
    )
    if linmix_train_excess is not None:
        collect_positive_log_y_bounds(excess_y_bounds, linmix_train_excess, linmix_train_excess_std)
        plotted_excess = plot_positive_excess_risk_curve(
            excess_ax,
            eps,
            linmix_train_excess,
            linmix_train_excess_std,
            color=colors['linmix'],
            marker=markers['linmix'],
            label=fr"Linear mixing [Lev et al. '25], {label_suffix}",
        ) or plotted_excess

    fast_linmix_train_excess, fast_linmix_train_excess_std = empirical_excess_risk_curve(
        base_record["train_mse_for_sigmas_fast"],
        base_record["train_mse_for_sigmas_fast_std"],
        base_record["baseline_train_mse_mean"],
    )
    if fast_linmix_train_excess is not None:
        collect_positive_log_y_bounds(excess_y_bounds, fast_linmix_train_excess, fast_linmix_train_excess_std)
        plotted_excess = plot_positive_excess_risk_curve(
            excess_ax,
            eps,
            fast_linmix_train_excess,
            fast_linmix_train_excess_std,
            color=colors['fast_linmix'],
            marker=markers['fast_linmix'],
            label=fr"Fast Linear mixing (ours), {label_suffix}",
        ) or plotted_excess

    ihm_train_excess, ihm_train_excess_std = empirical_excess_risk_curve(
        base_record["train_mse_for_sigmas_IHM"],
        base_record["train_mse_for_sigmas_std_IHM"],
        base_record["baseline_train_mse_mean"],
    )
    if ihm_train_excess is not None:
        collect_positive_log_y_bounds(excess_y_bounds, ihm_train_excess, ihm_train_excess_std)
        label_iterative = (
            fr"IHM [Lev et al. '26], {iters_IHM} iters, {label_suffix_iter}"
        )
        plotted_excess = plot_positive_excess_risk_curve(
            excess_ax,
            eps,
            ihm_train_excess,
            ihm_train_excess_std,
            linestyle='--',
            color=colors['hessian_mixing'],
            marker=markers['hessian_mixing'],
            label=label_iterative,
        ) or plotted_excess

    for fast_ihm_idx, plot_record in enumerate(fast_ihm_records):
        fast_ihm_train_excess, fast_ihm_train_excess_std = empirical_excess_risk_curve(
            plot_record["train_mse_for_sigmas_fast_IHM"],
            plot_record["train_mse_for_sigmas_fast_IHM_std"],
            plot_record["baseline_train_mse_mean"],
        )
        if fast_ihm_train_excess is None:
            continue

        collect_positive_log_y_bounds(excess_y_bounds, fast_ihm_train_excess, fast_ihm_train_excess_std)
        color = colors['fast_hessian_mixing']
        marker = markers['fast_hessian_mixing']
        if use_multi_fast_ihm_style:
            color = fast_ihm_colors[fast_ihm_idx % len(fast_ihm_colors)]
            marker = fast_ihm_markers[fast_ihm_idx % len(fast_ihm_markers)]

        label_iterative = (
            fr"Fast IHM (ours), {iters_IHM} iters, {label_suffix_iter}, "
            + plot_record["fast_ihm_inner_label_suffix"]
        )
        plotted_excess = plot_positive_excess_risk_curve(
            excess_ax,
            eps,
            fast_ihm_train_excess,
            fast_ihm_train_excess_std,
            linestyle='--',
            color=color,
            marker=marker,
            label=label_iterative,
        ) or plotted_excess

    if add_fast_ihs_baseline_horizontal_line:
        fast_ihs_baseline_train_mean = base_record.get("fast_ihs_baseline_train_mean")
        baseline_train_mse_mean = base_record.get("baseline_train_mse_mean")
        if fast_ihs_baseline_train_mean is not None and baseline_train_mse_mean is not None:
            fast_ihs_baseline_excess = float(fast_ihs_baseline_train_mean) - float(baseline_train_mse_mean)
            if np.isfinite(fast_ihs_baseline_excess) and fast_ihs_baseline_excess > 0.0:
                collect_positive_log_y_bounds(excess_y_bounds, [fast_ihs_baseline_excess])
                excess_ax.axhline(
                    fast_ihs_baseline_excess,
                    color='black',
                    linestyle='--',
                    linewidth=2.5,
                    label="Fast IHS baseline",
                    zorder=0,
                )
                plotted_excess = True

    if plotted_excess:
        excess_ax.set_xlabel(r"$\epsilon_{\mathrm{DP}}$", fontsize=38)
        excess_ax.set_ylabel("Excess Empirical Risk", fontsize=32)
        excess_ax.set_xscale('log')
        excess_ax.set_yscale('log')
        set_log_y_limits_from_bounds(excess_ax, excess_y_bounds)
        excess_ax.grid(True)
        set_log_y_ticks(excess_ax)

        excess_ax.tick_params(axis='both', which='both', labelsize=36)
        excess_ax.yaxis.get_offset_text().set_fontsize(36)
        plt.tight_layout()

        if add_legend_to_plot:
            plt.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, -0.02),
                bbox_transform=plt.gcf().transFigure,
                ncol=3,
                fontsize=22,
                frameon=False,
            )

        os.makedirs("plots", exist_ok=True)
        plot_file_token = base_record["fast_ihm_inner_file_token"]
        if use_multi_fast_ihm_style:
            plot_file_token = compact_filename_token(
                [record["fast_ihm_inner_file_token"] for record in plot_records],
                prefix="kf",
            )
        else:
            plot_file_token = compact_filename_token([plot_file_token], prefix="kf")
        plot_variant_token = fast_ihs_line_variant_file_token(add_fast_ihs_baseline_horizontal_line)
        plot_path = (
            f"plots/fmols_{dataset_type}_n{n}_d{d}_k{file_token_from_float(percentage_k_Mix)}_"
            f"{plot_file_token}_t{iters_IHM}_{eigval_noise}_{plot_variant_token}.pdf"
        )
        plt.savefig(filesystem_safe_output_path(plot_path), bbox_inches='tight')

    plt.close(excess_fig)

def save_empirical_excess_risk_plot(
    plot_records,
    *,
    dataset_type,
    n,
    d,
    percentage_k_Mix,
    percentage_k_IHM,
    iters_IHM,
    eigval_noise,
    epsilon_values,
    add_legend_to_plot,
    combine_fast_ihm_inner_curves,
):
    for add_fast_ihs_baseline_horizontal_line in (True, False):
        _save_empirical_excess_risk_plot_variant(
            plot_records,
            dataset_type=dataset_type,
            n=n,
            d=d,
            percentage_k_Mix=percentage_k_Mix,
            percentage_k_IHM=percentage_k_IHM,
            iters_IHM=iters_IHM,
            eigval_noise=eigval_noise,
            epsilon_values=epsilon_values,
            add_legend_to_plot=add_legend_to_plot,
            combine_fast_ihm_inner_curves=combine_fast_ihm_inner_curves,
            add_fast_ihs_baseline_horizontal_line=add_fast_ihs_baseline_horizontal_line,
        )

############# Hyperparameter T ############# 
iters_IHM = 3

############# Hyperparameter: k #############
# For IHM, k is going to be percentage_k_IHM * max(d, log(4T/varrho))
# For LinearMixing, k is going to be percentage_k_Mix * max(d, log(1/varrho))
percentage_k_IHM = 8.0
percentage_k_Mix = percentage_k_IHM * iters_IHM
percentage_k_IHM_fast = 6.0
# Fixed outer sketch size for the Fast IHS baseline.
# Unlike FastHessianMix, this stays independent of the fast-IHM outer sketch.
percentage_k_FastIHS_baseline = 8.0
# Optional inner ROS sketch size(s) for FastHessianMix.
# When a value is None, the inner sketch matches the outer sketch size k_val_FastIHM.
# This can be a single value or a list of values to run sequentially for each dataset.
percentage_k_FastIHM_inner = [4.0, 100.0, 200.0] # [4.0, 400.0, 800.0]

# Plotting mode for multiple FastHessianMix inner sketch sizes:
# 'combined' overlays all FastHessianMix inner-k curves on one plot per dataset.
# 'separate' preserves the current behavior of one plot per inner-k value.
fast_ihm_inner_plot_mode = 'combined'
if fast_ihm_inner_plot_mode not in {'combined', 'separate'}:
    raise ValueError("fast_ihm_inner_plot_mode must be either 'combined' or 'separate'")

# Number of Monte-Carlo iterations
iters = 20

transform = 'hadamard' # 'dct'
eigval_noise = 'Laplace' # choose between Gaussian, Laplace and internal (which is Gaussian) 

# Which methods to plot: pick from the next list 
# 'LinearMix'           : Linear Mixing
# 'Fast_LinearMix'      : Linear Mixing With Fast Sketch
# 'HessianMix'          : Hessian mixing with iters_IHM iterations
# 'Fast_HessianMix'     : Hessian mixing with iters_IHM iterations and Fast sketch 
# 'GaussianIHSBaseline' : Classical IHS with a Gaussian sketch and no privacy noise
# 'FastIHSBaseline'     : Classical fast IHS with the ROS sketch and no privacy noise
# 'GaussianSketchSolveBaseline' : Classical Gaussian sketch-and-solve over (X,y) with no privacy noise
# 'FastSketchSolveBaseline' : Legacy alias for the Gaussian sketch-and-solve baseline
# 'AdaSSP'              : AdaSSP baseline

methods_to_plot = ['HessianMix', 'FastHessianMix', 'FastIHSBaseline']

add_legend_to_plot = False 

# epsilon values to run our algorithm 
# epsilon_values = np.logspace(-0.5, 1.75, 5)  # np.logspace(-1.0, 1.0, 5)
epsilon_values = np.logspace(-0.5, 1.0, 6)

experiment_rng = np.random.default_rng()

os.makedirs("logs", exist_ok=True)
runtime_log_path = filesystem_safe_output_path(os.path.join("logs", "runtime_log.txt"))
with open(runtime_log_path, "w") as f:
    f.write("Fast Mixing OLS Runtime Log\n")
    f.write("===========================\n\n")

use_linmix = 'LinearMix' in methods_to_plot
use_fast_linmix = 'FastLinearMix' in methods_to_plot
use_ihm = 'HessianMix' in methods_to_plot
use_fast_ihm = 'FastHessianMix' in methods_to_plot
use_gaussian_ihs_baseline = 'GaussianIHSBaseline' in methods_to_plot
use_fast_ihs_baseline = 'FastIHSBaseline' in methods_to_plot
use_gaussian_sketch_solve_baseline = (
    'GaussianSketchSolveBaseline' in methods_to_plot
    or 'FastSketchSolveBaseline' in methods_to_plot
)
use_adassp = 'AdaSSP' in methods_to_plot

synthetic_ols_datasets = {
    'Gaussian',
    'correlated',
    'synthetic_Gaussian',
    'synthetic_correlated',
    'large_min_eigval',
}

# Full set of datasets to run our algorithms: can select a partial list of these for current run
# datasets_run_list = ['black_friday', 'years', 'beijing', \
#                    'rossman', 'airline', 'synthetic_correlated', 'large_min_eigval']

datasets_run_list = ['black_friday']

# Start looping over datasets 
for dataset_type in datasets_run_list:
    
    print('Running dataset ' + dataset_type)

    # Parse dataset 
    
    if dataset_type in synthetic_ols_datasets:
        n = 2**19
        d = 2**5
        C_max, X_train, X_test,\
            y_train, y_test, \
            lambda_min, lambda_max, lambda_min_XY, lambda_max_XY, _\
                = generate_synthetic_ols(n=n, d=d, type=dataset_type)
        dataset_title = dataset_titles.get(dataset_type, str(dataset_type))
        XY = np.hstack([X_train, y_train.reshape(-1, 1)])   # shape (n, d+1)
    else:
        C_max, X_train, y_train,\
            X_test, y_test, \
            lambda_min, lambda_max, lambda_min_XY, lambda_max_XY, \
            n, d, dataset_title = GetDataset(dataset_type)

    if transform == 'hadamard':
        n_train = X_train.shape[0]
        target_n = largest_power_of_two_leq(n_train)
        if target_n != n_train:
            subset_idx = np.sort(experiment_rng.choice(n_train, size=target_n, replace=False))
            X_train = X_train[subset_idx]
            y_train = y_train[subset_idx]
            print(
                "Hadamard mode: subsampled training set from n = "
                + str(n_train)
                + " to n = "
                + str(target_n)
            )

    n, d = X_train.shape
    X_train_T_local = X_train.T
    lambda_spectrum = np.linalg.eigvalsh(X_train_T_local @ X_train)
    lambda_min = float(np.min(lambda_spectrum))
    lambda_max = float(np.max(lambda_spectrum))
    XY = np.hstack([X_train, y_train.reshape(-1, 1)])   # shape (n, d+1)
    lambda_xy_spectrum = np.linalg.eigvalsh(XY.T @ XY)
    lambda_min_XY = float(np.min(lambda_xy_spectrum))
    lambda_max_XY = float(np.max(lambda_xy_spectrum))
    print('=============================')
    print('lambda min is ' + str(lambda_min)
          + ', lambda min XY is ' + str(lambda_min_XY))
    print('lambda max is ' + str(lambda_max)
          + ', lambda max XY is ' + str(lambda_max_XY))
    print('n is ' + str(n) + ', d is ' + str(d))
    print('=============================')

    # Delta and target varrho 
    delta_DP = 1/(n**2)
    target_varrho = delta_DP/10

    # Hyerparameter: tau - eigenvalue threshold constant 
    tau = np.sqrt(2.0 * np.log(np.max((1.5/delta_DP, 1.0/target_varrho))))
    tau_mix = np.sqrt(2.0 * np.log(np.max((2.0/delta_DP, 2.0/target_varrho))))
    
    # tau_fast_mix = np.sqrt(2.0 * np.log(np.max((2.5/delta_DP, 2.0/target_varrho))))
    tau_fast_mix = np.sqrt(2.0 * np.log(np.max((2.0/delta_DP, 2.0/target_varrho))))
        
    # Baseline: OLS
    ols_model = LinearRegression(fit_intercept=False)  # alpha is the regularization strength
    ols_model.fit(X_train, y_train)
    ridge_y_pred = ols_model.predict(X_test)
    baseline_test_mse = np.sum((y_test - ridge_y_pred)**2)
    ridge_y_pred_train = ols_model.predict(X_train)
    baseline_train_mse = np.sum((y_train - ridge_y_pred_train)**2)

    print('=============================')
    print('baseline test mse is  ' + str(baseline_test_mse/len(y_test)))
    print('baseline train mse is ' + str(baseline_train_mse/n))
    print('=============================')

    percentage_k_FastIHM_inner_values = normalize_fast_ihm_inner_percentages(percentage_k_FastIHM_inner)
    # Reuse the same starting RNG state for each inner-k sweep point so the
    # sweep is not biased by whichever configuration happens to run first.
    fast_ihm_inner_sweep_seed = int(
        experiment_rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64)
    )
    empirical_plot_records = []

    for percentage_k_FastIHM_inner_curr in percentage_k_FastIHM_inner_values:
        rng = np.random.default_rng(fast_ihm_inner_sweep_seed)
        if percentage_k_FastIHM_inner_curr is None:
            print('Running FastHessianMix inner sketch setting: outer')
        else:
            print('Running FastHessianMix inner sketch percentage ' + str(percentage_k_FastIHM_inner_curr))
        fast_ihm_inner_label_suffix, fast_ihm_inner_file_token = get_fast_ihm_inner_plot_metadata(
            percentage_k_FastIHM_inner_curr
        )

        # Set k_value 
        k_val_LinMix = np.max((int(percentage_k_Mix * d), int(percentage_k_Mix * np.log(2.0/target_varrho))))  # target_rho is the failure probability of the random projection
        k_val_IHM    = np.max((int(percentage_k_IHM * d), \
            int(percentage_k_IHM * np.log(4.0*iters_IHM/target_varrho))))  # target_rho is the failure probability of the random projection
        k_val_FastLinMix = k_val_LinMix 
        k_val_FastIHM    = np.max((int(percentage_k_IHM_fast * d), \
            int(percentage_k_IHM_fast * np.log(4.0*iters_IHM/target_varrho))))  # target_rho is the failure probability of the random projection
        k_val_FastIHS_baseline = np.max((
            int(percentage_k_FastIHS_baseline * d),
            int(percentage_k_FastIHS_baseline * np.log(4.0*iters_IHM/target_varrho)),
        ))
        if percentage_k_FastIHM_inner_curr is None:
            k_val_FastIHM_inner_curr = k_val_FastIHM
        else:
            k_val_FastIHM_inner_curr = np.max((int(percentage_k_FastIHM_inner_curr * d), \
            int(percentage_k_FastIHM_inner_curr * np.log(4.0*iters_IHM/target_varrho))))
            if k_val_FastIHM_inner_curr < 1:
                raise ValueError("k_val_FastIHM_inner must be positive")
            if k_val_FastIHM_inner_curr > n:
                raise ValueError("k_val_FastIHM_inner must be at most n")
    
        X_train_T = X_train.T
        baseline_test_mse_mean = baseline_test_mse / len(y_test)
        baseline_train_mse_mean = baseline_train_mse / n
        identity_d = np.eye(d)
        tri_upper_idx = np.triu_indices(d) if use_adassp else None
    
        # Prepare a list to store MSE for each sigma
        mse_for_sigmas              = [] if use_linmix else None
        mse_for_sigmas_std          = [] if use_linmix else None
        mse_for_sigmas_fast         = [] if use_fast_linmix else None
        mse_for_sigmas_fast_std     = [] if use_fast_linmix else None
        mse_for_sigmas_IHM          = [] if use_ihm else None
        mse_for_sigmas_std_IHM      = [] if use_ihm else None
        mse_for_sigmas_fast_IHM     = [] if use_fast_ihm else None
        mse_for_sigmas_std_fast_IHM = [] if use_fast_ihm else None
        mse_for_sigmas_adassp       = [] if use_adassp else None
        mse_for_sigmas_adassp_std   = [] if use_adassp else None
        
        train_mse_for_sigmas              = [] if use_linmix else None
        train_mse_for_sigmas_std          = [] if use_linmix else None
        train_mse_for_sigmas_IHM          = [] if use_ihm else None
        train_mse_for_sigmas_std_IHM      = [] if use_ihm else None
        train_mse_for_sigmas_fast         = [] if use_fast_linmix else None
        train_mse_for_sigmas_fast_std     = [] if use_fast_linmix else None
        train_mse_for_sigmas_fast_IHM     = [] if use_fast_ihm else None
        train_mse_for_sigmas_fast_IHM_std = [] if use_fast_ihm else None
        train_mse_for_sigmas_adassp       = [] if use_adassp else None
        train_mse_for_sigmas_adassp_std   = [] if use_adassp else None
    
        gaussian_ihs_baseline_test_mean = None
        gaussian_ihs_baseline_test_std = None
        gaussian_ihs_baseline_train_mean = None
        gaussian_ihs_baseline_train_std = None
        gaussian_ihs_baseline_time_mean = None
        fast_ihs_baseline_test_mean = None
        fast_ihs_baseline_test_std = None
        fast_ihs_baseline_train_mean = None
        fast_ihs_baseline_train_std = None
        fast_ihs_baseline_time_mean = None
        gaussian_sketch_solve_baseline_test_mean = None
        gaussian_sketch_solve_baseline_test_std = None
        gaussian_sketch_solve_baseline_train_mean = None
        gaussian_sketch_solve_baseline_train_std = None
        gaussian_sketch_solve_baseline_time_mean = None
        per_epsilon_runtime_log = []
        runtime_averages = {
            "LinearMix": [] if use_linmix else None,
            "FastLinearMix": [] if use_fast_linmix else None,
            "HessianMix": [] if use_ihm else None,
            "FastHessianMix": [] if use_fast_ihm else None,
            "AdaSSP": [] if use_adassp else None,
        }
    
        if use_gaussian_ihs_baseline:
            curr_test_mse_gaussian_IHS_baseline = np.empty(iters)
            curr_train_mse_gaussian_IHS_baseline = np.empty(iters)
            time_GaussianIHS_baseline = 0.0
    
            theta_gaussian_IHS_baseline = np.empty(d)
            residual_gaussian_IHS_baseline = np.empty(n)
            predicted_step_gaussian_IHS_baseline = np.empty_like(y_train)
            S_gaussian_IHS_baseline = np.empty((k_val_IHM, n))
            X_train_full_PR_hessian_gaussian_IHS_baseline = np.empty((k_val_IHM, d))
            H_hat_gaussian_IHS_baseline = np.empty((d, d))
            system_gaussian_IHS_baseline = np.empty((d, d))
            XY_Pr_gaussian_IHS_baseline = np.empty(d)
            gaussian_ihs_baseline_ridge = 1e-5
            gaussian_baseline_seed = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
            rng_gaussian_ihs_baseline = np.random.default_rng(gaussian_baseline_seed)
    
            for iter_idx in range(iters):
                theta_gaussian_IHS_baseline.fill(0.0)
                np.copyto(residual_gaussian_IHS_baseline, y_train)
                t0 = perf_counter_ns()
                for _ in range(iters_IHM):
                    fill_standard_normal(rng_gaussian_ihs_baseline, S_gaussian_IHS_baseline)
                    dot_into(S_gaussian_IHS_baseline, X_train, X_train_full_PR_hessian_gaussian_IHS_baseline)
    
                    dot_into(
                        X_train_full_PR_hessian_gaussian_IHS_baseline.T,
                        X_train_full_PR_hessian_gaussian_IHS_baseline,
                        H_hat_gaussian_IHS_baseline,
                    )
                    H_hat_gaussian_IHS_baseline /= k_val_IHM
                    np.copyto(system_gaussian_IHS_baseline, H_hat_gaussian_IHS_baseline)
                    system_gaussian_IHS_baseline += gaussian_ihs_baseline_ridge * identity_d
                    dot_into(X_train_T, residual_gaussian_IHS_baseline, XY_Pr_gaussian_IHS_baseline)
    
                    step = np.linalg.solve(system_gaussian_IHS_baseline, XY_Pr_gaussian_IHS_baseline)
                    theta_gaussian_IHS_baseline += step
                    dot_into(X_train, step, predicted_step_gaussian_IHS_baseline)
                    residual_gaussian_IHS_baseline -= predicted_step_gaussian_IHS_baseline
                t1 = perf_counter_ns()
                time_GaussianIHS_baseline += (t1 - t0)/1e9
    
                y_test_pred = X_test @ theta_gaussian_IHS_baseline
                curr_test_mse_gaussian_IHS_baseline[iter_idx] = (
                    np.mean((y_test - y_test_pred)**2)
                )
                y_train_pred = X_train @ theta_gaussian_IHS_baseline
                curr_train_mse_gaussian_IHS_baseline[iter_idx] = (
                    np.mean((y_train - y_train_pred)**2)
                )
    
            gaussian_ihs_baseline_test_mean = np.mean(curr_test_mse_gaussian_IHS_baseline)
            gaussian_ihs_baseline_test_std = 1.96 * np.std(curr_test_mse_gaussian_IHS_baseline) / np.sqrt(iters)
            gaussian_ihs_baseline_train_mean = np.mean(curr_train_mse_gaussian_IHS_baseline)
            gaussian_ihs_baseline_train_std = 1.96 * np.std(curr_train_mse_gaussian_IHS_baseline) / np.sqrt(iters)
            gaussian_ihs_baseline_time_mean = time_GaussianIHS_baseline / iters
    
        if use_fast_ihs_baseline:
            curr_test_mse_fast_IHS_baseline = np.empty(iters)
            curr_train_mse_fast_IHS_baseline = np.empty(iters)
            time_FastIHS_baseline = 0.0
    
            theta_fast_IHS_baseline = np.empty(d)
            residual_fast_IHS_baseline = np.empty(n)
            predicted_step_fast_IHS_baseline = np.empty_like(y_train)
            X_train_full_PR_hessian_fast_IHS_baseline = np.empty((k_val_FastIHS_baseline, d))
            H_hat_fast_IHS_baseline = np.empty((d, d))
            system_fast_IHS_baseline = np.empty((d, d))
            XY_Pr_fast_IHS_baseline = np.empty(d)
            fast_ihs_baseline_ridge = 1e-5
            baseline_seed = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
            rng_fast_ihs_baseline = np.random.default_rng(baseline_seed)
            Fdraw_fast_ihs_baseline = ROSMultiHDDraw(
                n=n,
                k=k_val_FastIHS_baseline,
                T=1,
                transform=transform,
                norm="ortho",
                rng=rng_fast_ihs_baseline,
            )
    
            for iter_idx in range(iters):
                theta_fast_IHS_baseline.fill(0.0)
                np.copyto(residual_fast_IHS_baseline, y_train)
                t0 = perf_counter_ns()
                for _ in range(iters_IHM):
                    Fdraw_fast_ihs_baseline.redraw()
                    np.copyto(
                        X_train_full_PR_hessian_fast_IHS_baseline,
                        Fdraw_fast_ihs_baseline.apply_F(X_train),
                    )
    
                    dot_into(
                        X_train_full_PR_hessian_fast_IHS_baseline.T,
                        X_train_full_PR_hessian_fast_IHS_baseline,
                        H_hat_fast_IHS_baseline,
                    )
                    np.copyto(system_fast_IHS_baseline, H_hat_fast_IHS_baseline)
                    system_fast_IHS_baseline += fast_ihs_baseline_ridge * identity_d
                    dot_into(X_train_T, residual_fast_IHS_baseline, XY_Pr_fast_IHS_baseline)
    
                    step = np.linalg.solve(system_fast_IHS_baseline, XY_Pr_fast_IHS_baseline)
                    theta_fast_IHS_baseline += step
                    dot_into(X_train, step, predicted_step_fast_IHS_baseline)
                    residual_fast_IHS_baseline -= predicted_step_fast_IHS_baseline
                t1 = perf_counter_ns()
                time_FastIHS_baseline += (t1 - t0)/1e9
    
                y_test_pred = X_test @ theta_fast_IHS_baseline
                curr_test_mse_fast_IHS_baseline[iter_idx] = (
                    np.mean((y_test - y_test_pred)**2)
                )
                y_train_pred = X_train @ theta_fast_IHS_baseline
                curr_train_mse_fast_IHS_baseline[iter_idx] = (
                    np.mean((y_train - y_train_pred)**2)
                )
    
            fast_ihs_baseline_test_mean = np.mean(curr_test_mse_fast_IHS_baseline)
            fast_ihs_baseline_test_std = 1.96 * np.std(curr_test_mse_fast_IHS_baseline) / np.sqrt(iters)
            fast_ihs_baseline_train_mean = np.mean(curr_train_mse_fast_IHS_baseline)
            fast_ihs_baseline_train_std = 1.96 * np.std(curr_train_mse_fast_IHS_baseline) / np.sqrt(iters)
            fast_ihs_baseline_time_mean = time_FastIHS_baseline / iters
    
        if use_gaussian_sketch_solve_baseline:
            curr_test_mse_gaussian_sketch_solve_baseline = np.empty(iters)
            curr_train_mse_gaussian_sketch_solve_baseline = np.empty(iters)
            time_GaussianSketchSolve_baseline = 0.0
    
            baseline_seed = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
            rng_gaussian_sketch_solve_baseline = np.random.default_rng(baseline_seed)
            S_gaussian_sketch_solve_baseline = np.empty((k_val_LinMix, n))
            XY_gaussian_sketch_solve_baseline = np.empty((k_val_LinMix, d + 1))
    
            for iter_idx in range(iters):
                t0 = perf_counter_ns()
                fill_standard_normal(rng_gaussian_sketch_solve_baseline, S_gaussian_sketch_solve_baseline)
                dot_into(S_gaussian_sketch_solve_baseline, XY, XY_gaussian_sketch_solve_baseline)
                theta_gaussian_sketch_solve_baseline, *_ = np.linalg.lstsq(
                    XY_gaussian_sketch_solve_baseline[:, :-1],
                    XY_gaussian_sketch_solve_baseline[:, -1],
                    rcond=None,
                )
                t1 = perf_counter_ns()
                time_GaussianSketchSolve_baseline += (t1 - t0)/1e9
    
                y_test_pred = X_test @ theta_gaussian_sketch_solve_baseline
                curr_test_mse_gaussian_sketch_solve_baseline[iter_idx] = (
                    np.mean((y_test - y_test_pred)**2)
                )
                y_train_pred = X_train @ theta_gaussian_sketch_solve_baseline
                curr_train_mse_gaussian_sketch_solve_baseline[iter_idx] = (
                    np.mean((y_train - y_train_pred)**2)
                )
    
            gaussian_sketch_solve_baseline_test_mean = np.mean(curr_test_mse_gaussian_sketch_solve_baseline)
            gaussian_sketch_solve_baseline_test_std = 1.96 * np.std(curr_test_mse_gaussian_sketch_solve_baseline) / np.sqrt(iters)
            gaussian_sketch_solve_baseline_train_mean = np.mean(curr_train_mse_gaussian_sketch_solve_baseline)
            gaussian_sketch_solve_baseline_train_std = 1.96 * np.std(curr_train_mse_gaussian_sketch_solve_baseline) / np.sqrt(iters)
            gaussian_sketch_solve_baseline_time_mean = time_GaussianSketchSolve_baseline / iters
    
        # Iterate over different epsilons
        for eps in tqdm(epsilon_values):
            print('Started eps = ' + str(eps))
            
            # Initiate arrays for results of current \eps
            curr_test_mse_linmix = np.empty(iters) if use_linmix else None
            curr_test_mse_fast_linmix = np.empty(iters) if use_fast_linmix else None
            curr_test_mse_IHM = np.empty(iters) if use_ihm else None
            curr_test_mse_fast_IHM = np.empty(iters) if use_fast_ihm else None
            curr_test_mse_adassp = np.empty(iters) if use_adassp else None
            
            curr_train_mse_linmix = np.empty(iters) if use_linmix else None
            curr_train_mse_fast_linmix = np.empty(iters) if use_fast_linmix else None
            curr_train_mse_IHM = np.empty(iters) if use_ihm else None
            curr_train_mse_fast_IHM = np.empty(iters) if use_fast_ihm else None
            curr_train_mse_adassp = np.empty(iters) if use_adassp else None
            
            ######################## Compute Noise Values ##########################
            # Baseline for numeric calculations: (\eps, \delta)-DP with classical Gaussian mechanism 
            sigma_eps_delta_DP_Gaussian = 2.0 * np.log(1.25/delta_DP) / (eps**2)
            
            if use_linmix:
                ######################## Linear Mixing ########################
                gamma_matrix = solve_gamma_renyi_full(
                    init_gamma=sigma_eps_delta_DP_Gaussian,
                    k=k_val_LinMix,
                    target_delta=delta_DP,
                    target_epsilon=eps,
                    inflation_norm=C_max**2,
                )
                sigma_eigenval = np.sqrt(1.0 + C_max**2) * gamma_matrix / np.sqrt(k_val_LinMix)
    
            if use_fast_linmix:
                ######################## Fast Linear Mixing ########################
                if eigval_noise == 'internal':
                    gamma_matrix_fast = solve_gamma_renyi_full_fast(
                        init_gamma=sigma_eps_delta_DP_Gaussian,
                        k=k_val_FastLinMix,
                        target_delta=delta_DP,
                        target_epsilon=eps,
                        inflation_norm=C_max**2,
                        add_eigval=True,
                    )
                    sigma_eigenval_fast = np.sqrt(1.0 + C_max**2) * gamma_matrix_fast / np.sqrt(k_val_LinMix)
                    tau_fast = np.sqrt(2.0 * np.log(np.max((1.5/delta_DP, 1.0/target_varrho))))
                else:
                    gamma_matrix_fast = solve_gamma_renyi_full_fast(
                        init_gamma=sigma_eps_delta_DP_Gaussian,
                        k=k_val_FastLinMix,
                        target_delta=delta_DP/3.0,
                        target_epsilon=eps/2.0,
                        inflation_norm=C_max**2,
                        add_eigval=False,
                    )
                    if eigval_noise == 'Gaussian':
                        sigma_eigenval_fast = np.sqrt(1.0 + C_max**2) * np.sqrt(2.0 * np.log(3.75/delta_DP)) / (eps/2.0)
                        tau_fast = np.sqrt(2.0 * np.log(np.max((1.5/delta_DP, 1.0/target_varrho))))
                    else:
                        sigma_eigenval_fast = np.sqrt(1.0 + C_max**2) / (eps/2.0)
                        tau_fast = np.max((np.log(1.5/delta_DP), np.sqrt(2.0 * np.log(1.0/target_varrho))))
    
            if use_ihm:
                ######################## IHM with T=iters_IHM ########################
                gamma_matrix_half_iter = solve_gamma_renyi_full_composition(
                    init_gamma=sigma_eps_delta_DP_Gaussian,
                    k=k_val_IHM,
                    target_delta=3.0*delta_DP/4.0,
                    target_epsilon=eps/2.0,
                    inflation_norm=0.0,
                    T=iters_IHM,
                )
                sigma_eigenval_half_iter = gamma_matrix_half_iter/np.sqrt(k_val_IHM)
                sigma_IHM_XTY = calibrateAnalyticGaussianMechanism(eps/2.0, delta_DP/4.0, C_max, tol=1.e-12)
                sigma_IHM_XTY *= np.sqrt(iters_IHM)
    
            if use_fast_ihm:
                ######################## Fast IHM with T=iters_IHM ########################
                if eigval_noise == 'internal':
                    gamma_matrix_half_iter_fast = solve_gamma_renyi_full_fast_composition(
                        init_gamma=sigma_eps_delta_DP_Gaussian,
                        k=k_val_FastIHM,
                        target_delta=3.0*delta_DP/4.0,
                        target_epsilon=eps/2.0,
                        inflation_norm=0.0,
                        T=iters_IHM,
                        add_eigval=True,
                    )
                    sigma_eigenval_half_iter_fast = gamma_matrix_half_iter_fast/np.sqrt(k_val_FastIHM)
                    tau_fast_mix = np.max((np.log(3.0*iters_IHM/delta_DP/2.0), np.log(16.0*iters_IHM/target_varrho)))
                else:
                    gamma_matrix_half_iter_fast = solve_gamma_renyi_full_fast_composition(
                        init_gamma=sigma_eps_delta_DP_Gaussian,
                        k=k_val_FastIHM,
                        target_delta=delta_DP/3.0,
                        target_epsilon=eps/3.0,
                        inflation_norm=0.0,
                        T=iters_IHM,
                        add_eigval=False,
                    )
                    if eigval_noise == 'Gaussian':
                        sigma_eigenval_half_iter_fast = np.sqrt(2.0 * np.log(5.0/delta_DP))/(eps/4.0)
                        tau_fast_mix = np.sqrt(2.0 * np.log(np.max((2.0/delta_DP, 1.0/target_varrho))))
                    else:
                        sigma_eigenval_half_iter_fast = 6.0*iters_IHM/(eps)
                        tau_fast_mix = np.max((np.log(3.0*iters_IHM/delta_DP), np.log(16.0*iters_IHM/target_varrho)))
    
                sigma_IHM_XTY_fast = calibrateAnalyticGaussianMechanism(eps/3.0, delta_DP/3.0, C_max, tol=1.e-12)
                sigma_IHM_XTY_fast *= np.sqrt(iters_IHM)
    
            if use_adassp:
                ######################## AdaSSP ########################
                sigma_base_AdaSSP_XTX = calibrateAnalyticGaussianMechanism(eps/3.0, delta_DP/3.0, 1.0, tol=1.e-12)
                sigma_base_AdaSSP_XTY = calibrateAnalyticGaussianMechanism(eps/3.0, delta_DP/3.0, C_max, tol=1.e-12)
            
            # Monte-carlo run start here 
            time_classical   = 0.0 if use_linmix else None
            time_fast        = 0.0 if use_fast_linmix else None
            time_HessMix     = 0.0 if use_ihm else None
            time_FastHessMix = 0.0 if use_fast_ihm else None
            time_AdaSSP      = 0.0 if use_adassp else None
    
            if use_linmix:
                S_linmix = np.empty((k_val_LinMix, n))
                N_linmix = np.empty((k_val_LinMix, d))
                N_linmix_y = np.empty(k_val_LinMix)
                X_train_full_PR_linmix = np.empty((k_val_LinMix, d))
                y_PR_linmix = np.empty(k_val_LinMix)
    
            if use_fast_linmix:
                S_g_fast_linmix = np.empty((k_val_LinMix, k_val_LinMix))
                N_fast_linmix = np.empty((k_val_LinMix, d))
                N_fast_linmix_y = np.empty(k_val_LinMix)
                S_XY_fast_linmix = np.empty((k_val_LinMix, d + 1))
                X_train_full_PR_fast_linmix = np.empty((k_val_LinMix, d))
                y_PR_fast_linmix = np.empty(k_val_LinMix)
                Fdraw_linmix = ROSMultiHDDraw(
                    n=n,
                    k=k_val_LinMix,
                    T=1,
                    transform=transform,
                    norm="ortho",
                    rng=rng,
                )
    
            if use_ihm:
                theta_IHM_T_iter = np.empty(d)
                residual_IHM = np.empty(n)
                clipped_residual_IHM = np.empty_like(y_train)
                predicted_step_IHM = np.empty_like(y_train)
                S_ihm = np.empty((k_val_IHM, n))
                N_ihm = np.empty((k_val_IHM, d))
                X_train_full_PR_hessian_IHM = np.empty((k_val_IHM, d))
                rhs_noise_IHM = np.empty(d)
                H_hat_IHM = np.empty((d, d))
                XY_Pr_IHM = np.empty(d)
    
            if use_fast_ihm:
                theta_fast_IHM_T_iter = np.empty(d)
                residual_fast_IHM = np.empty(n)
                clipped_residual_fast_IHM = np.empty_like(y_train)
                predicted_step_fast_IHM = np.empty_like(y_train)
                S_g_fast_ihm = np.empty((k_val_FastIHM, k_val_FastIHM_inner_curr))
                N_fast_ihm = np.empty((k_val_FastIHM, d))
                S_X_fast_ihm = np.empty((k_val_FastIHM, d))
                X_train_full_PR_hessian_fast_IHM = np.empty((k_val_FastIHM, d))
                rhs_noise_fast_IHM = np.empty(d)
                H_hat_fast_IHM = np.empty((d, d))
                XY_Pr_fast_IHM = np.empty(d)
                Fdraw_fast_ihm = ROSMultiHDDraw(
                    n=n,
                    k=k_val_FastIHM_inner_curr,
                    T=1,
                    transform=transform,
                    norm="ortho",
                    rng=rng,
                )
    
            if use_adassp:
                XTX_adassp = np.empty((d, d))
                XTY_adassp = np.empty(d)
                N_upper_adassp = np.empty((d, d))
                N_sym_adassp = np.empty((d, d))
                XTX_noisy_adassp = np.empty((d, d))
                system_adassp = np.empty((d, d))
                XTY_noise_adassp = np.empty(d)
                XTY_noisy_adassp = np.empty(d)
            
            for iter_idx in tqdm(range(iters)):
                if use_linmix:
                    t0 = perf_counter_ns()
                    fill_standard_normal(rng, S_linmix)
                    fill_standard_normal(rng, N_linmix)
                    fill_standard_normal(rng, N_linmix_y)
    
                    gamma_tilde = np.max((0, lambda_min_XY - sigma_eigenval * (tau - rng.standard_normal())))
                    gamma_tilde = np.sqrt(np.max((0, gamma_matrix - gamma_tilde)))
    
                    dot_into(S_linmix, X_train, X_train_full_PR_linmix)
                    X_train_full_PR_linmix += gamma_tilde * N_linmix
                    dot_into(S_linmix, y_train, y_PR_linmix)
                    y_PR_linmix += gamma_tilde * N_linmix_y
    
                    theta_LinMix, *_ = np.linalg.lstsq(X_train_full_PR_linmix, y_PR_linmix, rcond=None)
                    t1 = perf_counter_ns()
                    time_classical += (t1 - t0)/1e9
    
                if use_fast_linmix:
                    t0 = perf_counter_ns()
                    fill_standard_normal(rng, S_g_fast_linmix)
                    fill_standard_normal(rng, N_fast_linmix)
                    fill_standard_normal(rng, N_fast_linmix_y)
                    Fdraw_linmix.redraw()
                    m_hat_XY, max_offdiag_ftf_XY, SRFT_XY = max_norm_Xt_FtF_minus_I_ei(XY, Fdraw_linmix)
                    lambda_min_tilde_XY = np.min(np.linalg.eigvalsh(SRFT_XY.T @ SRFT_XY))
                    sigma_tilde = direct_private_sigma_tilde(
                        gamma=gamma_matrix_fast,
                        m_hat=m_hat_XY,
                        delta_hat=max_offdiag_ftf_XY,
                        lambda_hat=lambda_min_tilde_XY,
                        eta=sigma_eigenval_fast,
                        tau=tau_fast,
                        rng=rng,
                    )
    
                    dot_into(S_g_fast_linmix, SRFT_XY, S_XY_fast_linmix)
                    np.copyto(X_train_full_PR_fast_linmix, S_XY_fast_linmix[:, :-1])
                    X_train_full_PR_fast_linmix += sigma_tilde * N_fast_linmix
                    np.copyto(y_PR_fast_linmix, S_XY_fast_linmix[:, -1])
                    y_PR_fast_linmix += sigma_tilde * N_fast_linmix_y
    
                    theta_fast_LinMix, *_ = np.linalg.lstsq(X_train_full_PR_fast_linmix, y_PR_fast_linmix, rcond=None)
                    t1 = perf_counter_ns()
                    time_fast += (t1 - t0)/1e9
    
                if use_ihm:
                    gamma_tilde = np.max((0, lambda_min - sigma_eigenval_half_iter * (tau_mix - rng.standard_normal())))
                    gamma_final = np.sqrt(np.max((0, gamma_matrix_half_iter - gamma_tilde)))
    
                    theta_IHM_T_iter.fill(0.0)
                    np.copyto(residual_IHM, y_train)
                    t0 = perf_counter_ns()
                    for iter_hess in range(iters_IHM):
                        fill_standard_normal(rng, S_ihm)
                        fill_standard_normal(rng, N_ihm)
                        fill_standard_normal(rng, rhs_noise_IHM)
                        dot_into(S_ihm, X_train, X_train_full_PR_hessian_IHM)
                        X_train_full_PR_hessian_IHM += gamma_final * N_ihm
    
                        dot_into(X_train_full_PR_hessian_IHM.T, X_train_full_PR_hessian_IHM, H_hat_IHM)
                        H_hat_IHM /= k_val_IHM
                        np.clip(residual_IHM, -C_max, C_max, out=clipped_residual_IHM)
                        dot_into(X_train_T, clipped_residual_IHM, XY_Pr_IHM)
                        XY_Pr_IHM += sigma_IHM_XTY * rhs_noise_IHM
    
                        step = np.linalg.solve(H_hat_IHM, XY_Pr_IHM)
                        theta_IHM_T_iter += step
                        dot_into(X_train, step, predicted_step_IHM)
                        residual_IHM -= predicted_step_IHM
                    t1 = perf_counter_ns()
                    time_HessMix += (t1 - t0)/1e9
    
                if use_fast_ihm:
                    theta_fast_IHM_T_iter.fill(0.0)
                    np.copyto(residual_fast_IHM, y_train)
                    t0 = perf_counter_ns()
                    for iter_hess in range(iters_IHM):
                        fill_standard_normal(rng, S_g_fast_ihm)
                        fill_standard_normal(rng, N_fast_ihm)
                        fill_standard_normal(rng, rhs_noise_fast_IHM)
    
                        Fdraw_fast_ihm.redraw()
                        m_hat, max_offdiag_ftf_X, SRFT_X = max_norm_Xt_FtF_minus_I_ei(X_train, Fdraw_fast_ihm)
                        lambda_min_tilde_X = np.min(np.linalg.eigvalsh(SRFT_X.T @ SRFT_X))
                        sigma_tilde = direct_private_sigma_tilde(
                            gamma=gamma_matrix_half_iter_fast,
                            m_hat=m_hat,
                            delta_hat=max_offdiag_ftf_X,
                            lambda_hat=lambda_min_tilde_X,
                            eta=sigma_eigenval_half_iter_fast,
                            tau=tau_fast_mix,
                            rng=rng,
                        )
    
                        dot_into(S_g_fast_ihm, SRFT_X, S_X_fast_ihm)
                        np.copyto(X_train_full_PR_hessian_fast_IHM, S_X_fast_ihm)
                        X_train_full_PR_hessian_fast_IHM += sigma_tilde * N_fast_ihm
    
                        dot_into(X_train_full_PR_hessian_fast_IHM.T, X_train_full_PR_hessian_fast_IHM, H_hat_fast_IHM)
                        H_hat_fast_IHM /= k_val_FastIHM
                        np.clip(residual_fast_IHM, -C_max, C_max, out=clipped_residual_fast_IHM)
                        dot_into(X_train_T, clipped_residual_fast_IHM, XY_Pr_fast_IHM)
                        XY_Pr_fast_IHM += sigma_IHM_XTY_fast * rhs_noise_fast_IHM
    
                        step = np.linalg.solve(H_hat_fast_IHM, XY_Pr_fast_IHM)
                        theta_fast_IHM_T_iter += step
                        dot_into(X_train, step, predicted_step_fast_IHM)
                        residual_fast_IHM -= predicted_step_fast_IHM
                    t1 = perf_counter_ns()
                    time_FastHessMix += (t1 - t0)/1e9
    
                if use_adassp:
                    t0 = perf_counter_ns()
                    lambda_min_tilde = np.max((
                        0.0,
                        lambda_min + sigma_base_AdaSSP_XTX * rng.standard_normal() - sigma_base_AdaSSP_XTX**2,
                    ))
                    lambda_adassp = np.max((
                        0.0,
                        np.sqrt(d * np.log(2.0 * (d**2) / target_varrho)) * sigma_base_AdaSSP_XTX - lambda_min_tilde,
                    ))
    
                    dot_into(X_train_T, X_train, XTX_adassp)
                    dot_into(X_train_T, y_train, XTY_adassp)
                    fill_standard_normal(rng, N_upper_adassp)
                    N_sym_adassp.fill(0.0)
                    N_sym_adassp[tri_upper_idx] = N_upper_adassp[tri_upper_idx]
                    N_sym_adassp[(tri_upper_idx[1], tri_upper_idx[0])] = N_upper_adassp[tri_upper_idx]
                    np.copyto(XTX_noisy_adassp, XTX_adassp)
                    XTX_noisy_adassp += sigma_base_AdaSSP_XTX * N_sym_adassp
                    fill_standard_normal(rng, XTY_noise_adassp)
                    np.copyto(XTY_noisy_adassp, XTY_adassp)
                    XTY_noisy_adassp += sigma_base_AdaSSP_XTY * XTY_noise_adassp
                    np.copyto(system_adassp, XTX_noisy_adassp)
                    system_adassp.flat[::d + 1] += lambda_adassp
                    theta_adassp = np.linalg.solve(system_adassp, XTY_noisy_adassp)
                    t1 = perf_counter_ns()
                    time_AdaSSP += (t1 - t0)/1e9
                
                ############################### Evaluate ###############################
                # Evaluate on original test data
                if use_linmix:
                    y_test_pred = X_test @ theta_LinMix
                    curr_test_mse_linmix[iter_idx] = np.mean((y_test - y_test_pred)**2)
                    y_train_pred = X_train @ theta_LinMix
                    curr_train_mse_linmix[iter_idx] = np.mean((y_train - y_train_pred)**2)
    
                if use_ihm:
                    y_test_pred = X_test @ theta_IHM_T_iter
                    curr_test_mse_IHM[iter_idx] = np.mean((y_test - y_test_pred)**2)
                    y_train_pred = X_train @ theta_IHM_T_iter
                    curr_train_mse_IHM[iter_idx] = np.mean((y_train - y_train_pred)**2)
    
                if use_fast_linmix:
                    y_test_pred = X_test @ theta_fast_LinMix
                    curr_test_mse_fast_linmix[iter_idx] = np.mean((y_test - y_test_pred)**2)
                    y_train_pred = X_train @ theta_fast_LinMix
                    curr_train_mse_fast_linmix[iter_idx] = np.mean((y_train - y_train_pred)**2)
    
                if use_fast_ihm:
                    y_test_pred = X_test @ theta_fast_IHM_T_iter
                    curr_test_mse_fast_IHM[iter_idx] = np.mean((y_test - y_test_pred)**2)
                    y_train_pred = X_train @ theta_fast_IHM_T_iter
                    curr_train_mse_fast_IHM[iter_idx] = np.mean((y_train - y_train_pred)**2)
    
                if use_adassp:
                    y_test_pred = X_test @ theta_adassp
                    curr_test_mse_adassp[iter_idx] = np.mean((y_test - y_test_pred)**2)
                    y_train_pred = X_train @ theta_adassp
                    curr_train_mse_adassp[iter_idx] = np.mean((y_train - y_train_pred)**2)
    
            # MSEs and confidence intervals
            if use_linmix:
                mse_for_sigmas.append(np.mean(curr_test_mse_linmix))
                mse_for_sigmas_std.append(1.96 * np.std(curr_test_mse_linmix)/np.sqrt(iters))
                train_mse_for_sigmas.append(np.mean(curr_train_mse_linmix))
                train_mse_for_sigmas_std.append(1.96 * np.std(curr_train_mse_linmix)/np.sqrt(iters))
            if use_fast_linmix:
                mse_for_sigmas_fast.append(np.mean(curr_test_mse_fast_linmix))
                mse_for_sigmas_fast_std.append(1.96 * np.std(curr_test_mse_fast_linmix)/np.sqrt(iters))
                train_mse_for_sigmas_fast.append(np.mean(curr_train_mse_fast_linmix))
                train_mse_for_sigmas_fast_std.append(1.96 * np.std(curr_train_mse_fast_linmix)/np.sqrt(iters))
            if use_ihm:
                mse_for_sigmas_IHM.append(np.mean(curr_test_mse_IHM))
                mse_for_sigmas_std_IHM.append(1.96 * np.std(curr_test_mse_IHM)/np.sqrt(iters))
                train_mse_for_sigmas_IHM.append(np.mean(curr_train_mse_IHM))
                train_mse_for_sigmas_std_IHM.append(1.96 * np.std(curr_train_mse_IHM)/np.sqrt(iters))
            if use_fast_ihm:
                mse_for_sigmas_fast_IHM.append(np.mean(curr_test_mse_fast_IHM))
                mse_for_sigmas_std_fast_IHM.append(1.96 * np.std(curr_test_mse_fast_IHM)/np.sqrt(iters))
                train_mse_for_sigmas_fast_IHM.append(np.mean(curr_train_mse_fast_IHM))
                train_mse_for_sigmas_fast_IHM_std.append(1.96 * np.std(curr_train_mse_fast_IHM)/np.sqrt(iters))
                
            if use_adassp:
                mse_for_sigmas_adassp.append(np.mean(curr_test_mse_adassp))
                mse_for_sigmas_adassp_std.append(1.96 * np.std(curr_test_mse_adassp)/np.sqrt(iters))
                train_mse_for_sigmas_adassp.append(np.mean(curr_train_mse_adassp))
                train_mse_for_sigmas_adassp_std.append(1.96 * np.std(curr_train_mse_adassp)/np.sqrt(iters))
    
            runtime_results_eps = {}
            if use_linmix:
                runtime_results_eps["LinearMix"] = time_classical / iters
                runtime_averages["LinearMix"].append(runtime_results_eps["LinearMix"])
            if use_fast_linmix:
                runtime_results_eps["FastLinearMix"] = time_fast / iters
                runtime_averages["FastLinearMix"].append(runtime_results_eps["FastLinearMix"])
            if use_ihm:
                runtime_results_eps["HessianMix"] = time_HessMix / iters
                runtime_averages["HessianMix"].append(runtime_results_eps["HessianMix"])
            if use_fast_ihm:
                runtime_results_eps["FastHessianMix"] = time_FastHessMix / iters
                runtime_averages["FastHessianMix"].append(runtime_results_eps["FastHessianMix"])
            if use_adassp:
                runtime_results_eps["AdaSSP"] = time_AdaSSP / iters
                runtime_averages["AdaSSP"].append(runtime_results_eps["AdaSSP"])
            per_epsilon_runtime_log.append((float(eps), runtime_results_eps))
    
            print('====================')
            if use_linmix:
                print('Test Risk Sketch and Solve: '      + str(np.mean(curr_test_mse_linmix)))
            if use_fast_linmix:
                print('Test Risk Fast Sketch and Solve: ' + str(np.mean(curr_test_mse_fast_linmix)))
            if use_ihm:
                print('Test Risk Hessian ' + str(iters_IHM) + ' iters: ' + str(np.mean(curr_test_mse_IHM)))
            if use_fast_ihm:
                print('Test Risk Fast Hessian ' + str(iters_IHM) + ' iters: ' + str(np.mean(curr_test_mse_fast_IHM)))
            if use_gaussian_ihs_baseline:
                print('Test Risk Gaussian IHS baseline ' + str(iters_IHM) + ' iters: ' + str(gaussian_ihs_baseline_test_mean))
            if use_fast_ihs_baseline:
                print('Test Risk Fast IHS baseline ' + str(iters_IHM) + ' iters: ' + str(fast_ihs_baseline_test_mean))
            if use_gaussian_sketch_solve_baseline:
                print('Test Risk Gaussian sketch-and-solve baseline: ' + str(gaussian_sketch_solve_baseline_test_mean))
            if use_adassp:
                print('Test Risk AdaSSP: ' + str(np.mean(curr_test_mse_adassp)))
    
            print('====================')
            if use_linmix:
                print('Empirical Risk Sketch and Solve: '      + str(np.mean(curr_train_mse_linmix)))
            if use_fast_linmix:
                print('Empirical Risk Fast Sketch and Solve: ' + str(np.mean(curr_train_mse_fast_linmix)))
            if use_ihm:
                print('Empirical Risk Hessian: ' + str(np.mean(curr_train_mse_IHM)))
            if use_fast_ihm:
                print('Empirical Risk Fast Hessian ' + str(iters_IHM) + ' iters: ' + str(np.mean(curr_train_mse_fast_IHM)))
            if use_gaussian_ihs_baseline:
                print('Empirical Risk Gaussian IHS baseline ' + str(iters_IHM) + ' iters: ' + str(gaussian_ihs_baseline_train_mean))
            if use_fast_ihs_baseline:
                print('Empirical Risk Fast IHS baseline ' + str(iters_IHM) + ' iters: ' + str(fast_ihs_baseline_train_mean))
            if use_gaussian_sketch_solve_baseline:
                print('Empirical Risk Gaussian sketch-and-solve baseline: ' + str(gaussian_sketch_solve_baseline_train_mean))
            if use_adassp:
                print('Empirical Risk AdaSSP: ' + str(np.mean(curr_train_mse_adassp)))
    
            print('====================')
            if use_linmix:
                print("Time Classical: " + str(time_classical/iters))
            if use_fast_linmix:
                print("Time Fast: " + str(time_fast/iters))
            if use_ihm:
                print("Time IHM: " + str(time_HessMix/iters))
            if use_fast_ihm:
                print("Time Fast IHM: " + str(time_FastHessMix/iters))
            if use_gaussian_ihs_baseline:
                print("Time Gaussian IHS baseline: " + str(gaussian_ihs_baseline_time_mean))
            if use_fast_ihs_baseline:
                print("Time Fast IHS baseline: " + str(fast_ihs_baseline_time_mean))
            if use_gaussian_sketch_solve_baseline:
                print("Time Gaussian sketch-and-solve baseline: " + str(gaussian_sketch_solve_baseline_time_mean))
            if use_adassp:
                print("Time AdaSSP: " + str(time_AdaSSP/iters))
            print('====================')

        overall_runtime_results = {}
        for method_name, values in runtime_averages.items():
            if values is not None and len(values) > 0:
                overall_runtime_results[method_name] = float(np.mean(values))
        if use_gaussian_ihs_baseline:
            overall_runtime_results["GaussianIHSBaseline"] = float(gaussian_ihs_baseline_time_mean)
        if use_fast_ihs_baseline:
            overall_runtime_results["FastIHSBaseline"] = float(fast_ihs_baseline_time_mean)
        if use_gaussian_sketch_solve_baseline:
            overall_runtime_results["GaussianSketchSolveBaseline"] = float(gaussian_sketch_solve_baseline_time_mean)
        
        current_plot_record = {
            "fast_ihm_inner_label_suffix": fast_ihm_inner_label_suffix,
            "fast_ihm_inner_file_token": fast_ihm_inner_file_token,
            "fast_ihm_inner_percentage": None if percentage_k_FastIHM_inner_curr is None else float(percentage_k_FastIHM_inner_curr),
            "overall_runtime_by_method": overall_runtime_results.copy(),
            "baseline_train_mse_mean": float(baseline_train_mse_mean),
            "train_mse_for_sigmas": None if train_mse_for_sigmas is None else np.asarray(train_mse_for_sigmas, dtype=float).copy(),
            "train_mse_for_sigmas_std": None if train_mse_for_sigmas_std is None else np.asarray(train_mse_for_sigmas_std, dtype=float).copy(),
            "train_mse_for_sigmas_fast": None if train_mse_for_sigmas_fast is None else np.asarray(train_mse_for_sigmas_fast, dtype=float).copy(),
            "train_mse_for_sigmas_fast_std": None if train_mse_for_sigmas_fast_std is None else np.asarray(train_mse_for_sigmas_fast_std, dtype=float).copy(),
            "train_mse_for_sigmas_IHM": None if train_mse_for_sigmas_IHM is None else np.asarray(train_mse_for_sigmas_IHM, dtype=float).copy(),
            "train_mse_for_sigmas_std_IHM": None if train_mse_for_sigmas_std_IHM is None else np.asarray(train_mse_for_sigmas_std_IHM, dtype=float).copy(),
            "train_mse_for_sigmas_fast_IHM": None if train_mse_for_sigmas_fast_IHM is None else np.asarray(train_mse_for_sigmas_fast_IHM, dtype=float).copy(),
            "train_mse_for_sigmas_fast_IHM_std": None if train_mse_for_sigmas_fast_IHM_std is None else np.asarray(train_mse_for_sigmas_fast_IHM_std, dtype=float).copy(),
            "gaussian_sketch_solve_baseline_train_mean": gaussian_sketch_solve_baseline_train_mean,
            "gaussian_ihs_baseline_train_mean": gaussian_ihs_baseline_train_mean,
            "fast_ihs_baseline_train_mean": fast_ihs_baseline_train_mean,
        }
        empirical_plot_records.append(current_plot_record)
        if fast_ihm_inner_plot_mode == 'separate':
            save_empirical_excess_risk_plot(
                [current_plot_record],
                dataset_type=dataset_type,
                n=n,
                d=d,
                percentage_k_Mix=percentage_k_Mix,
                percentage_k_IHM=percentage_k_IHM,
                iters_IHM=iters_IHM,
                eigval_noise=eigval_noise,
                epsilon_values=epsilon_values,
                add_legend_to_plot=add_legend_to_plot,
                combine_fast_ihm_inner_curves=False,
            )
    
        with open(runtime_log_path, "a") as f:
            f.write(f"Dataset: {dataset_type}\n")
            f.write(f"percentage_k_FastIHS_baseline = {percentage_k_FastIHS_baseline:.1f}\n")
            if percentage_k_FastIHM_inner_curr is None:
                f.write("percentage_k_FastIHM_inner = outer\n")
            else:
                f.write(f"percentage_k_FastIHM_inner = {percentage_k_FastIHM_inner_curr:.1f}\n")
            f.write(
                f"n = {n}, d = {d}, MonteCarlo iters = {iters}, IHM iters = {iters_IHM}, "
                f"transform = {transform}, eigval_noise = {eigval_noise}\n"
            )
            f.write("Per-epsilon average runtime (seconds):\n")
            for eps_value, runtime_results_eps in per_epsilon_runtime_log:
                f.write(f"  epsilon = {eps_value:.10g}\n")
                for method_name, value in runtime_results_eps.items():
                    f.write(f"    {method_name}: {value:.10f}\n")
            if use_gaussian_ihs_baseline or use_fast_ihs_baseline or use_gaussian_sketch_solve_baseline:
                f.write("Epsilon-independent average runtime (seconds):\n")
            if use_gaussian_ihs_baseline:
                f.write(f"  GaussianIHSBaseline: {gaussian_ihs_baseline_time_mean:.10f}\n")
            if use_fast_ihs_baseline:
                f.write(f"  FastIHSBaseline: {fast_ihs_baseline_time_mean:.10f}\n")
            if use_gaussian_sketch_solve_baseline:
                f.write(f"  GaussianSketchSolveBaseline: {gaussian_sketch_solve_baseline_time_mean:.10f}\n")
            f.write("Average runtime across all epsilons (seconds):\n")
            for method_name, value in overall_runtime_results.items():
                f.write(f"  {method_name}: {value:.10f}\n")
            f.write("\n")
        
        # Save times only
        timing_results = {}
        if use_gaussian_sketch_solve_baseline:
            timing_results["Time Classical"] = overall_runtime_results["GaussianSketchSolveBaseline"]
        elif use_linmix:
            timing_results["Time Classical"] = overall_runtime_results["LinearMix"]
        if use_fast_linmix:
            timing_results["Time Fast"] = overall_runtime_results["FastLinearMix"]
        if use_ihm:
            timing_results["Time IHS"] = overall_runtime_results["HessianMix"]
        if use_fast_ihm:
            timing_results["Time Fast IHS"] = overall_runtime_results["FastHessianMix"]
        if use_gaussian_ihs_baseline:
            timing_results["Time Gaussian IHS baseline"] = overall_runtime_results["GaussianIHSBaseline"]
        if use_fast_ihs_baseline:
            timing_results["Time Fast IHS baseline"] = overall_runtime_results["FastIHSBaseline"]
        if use_gaussian_sketch_solve_baseline and "Time Classical" not in timing_results:
            timing_results["Time Gaussian sketch-and-solve baseline"] = overall_runtime_results["GaussianSketchSolveBaseline"]
        if use_adassp:
            timing_results["Time AdaSSP"] = overall_runtime_results["AdaSSP"]
    
        os.makedirs("timings", exist_ok=True)
        timing_inner_file_token = compact_filename_token(
            [fast_ihm_inner_file_token],
            prefix="kf",
        )
        timings_log_path = os.path.join(
            "timings",
            f"tmols_{dataset_type}_{timing_inner_file_token}.txt",
        )
        with open(filesystem_safe_output_path(timings_log_path), "w") as f:
            f.write(f"percentage_k_FastIHS_baseline: {percentage_k_FastIHS_baseline:.1f}\n")
            if percentage_k_FastIHM_inner_curr is None:
                f.write("percentage_k_FastIHM_inner: outer\n")
            else:
                f.write(f"percentage_k_FastIHM_inner: {percentage_k_FastIHM_inner_curr:.1f}\n")
            for name, val in timing_results.items():
                f.write(f"{name}: {val}\n")

    save_speedup_summary(
        empirical_plot_records,
        dataset_type=dataset_type,
        percentage_k_FastIHS_baseline=percentage_k_FastIHS_baseline,
        iters_IHM=iters_IHM,
        eigval_noise=eigval_noise,
    )

    if fast_ihm_inner_plot_mode == 'combined':
        save_empirical_excess_risk_plot(
            empirical_plot_records,
            dataset_type=dataset_type,
            n=n,
            d=d,
            percentage_k_Mix=percentage_k_Mix,
            percentage_k_IHM=percentage_k_IHM,
            iters_IHM=iters_IHM,
            eigval_noise=eigval_noise,
            epsilon_values=epsilon_values,
            add_legend_to_plot=add_legend_to_plot,
            combine_fast_ihm_inner_curves=True,
        )
