from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.floating]


def average_rate_component(
    regulated_rates: Array,                 # shape: (T, C)
    sales_by_customer_class: Array,         # shape: (T, C)
) -> Array:
    """Compute average rate component by period.

    Formula:
    RC_t = (sum_c r_{t,c} * Q_{t,c}) / (sum_c Q_{t,c}) = TotalRevenue_t / TotalSales_t

    Args:
        regulated_rates: Array of shape (T, C) with rates by customer class.
        sales_by_customer_class: Array of shape (T, C) with sales volumes by
            customer class.

    Returns:
        Array of shape (T,) with the average rate component per period.
    """
    total_revenue_t = (regulated_rates * sales_by_customer_class).sum(axis=1)
    total_sales_t = sales_by_customer_class.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rc_t = np.where(total_sales_t > 0, total_revenue_t / total_sales_t, 0.0)
    return rc_t


def rate_volatility(
    regulated_rates: Array,                 # shape: (T, C)
    class_weights: Array,                   # shape: (C,)
    previous_year_rates: Optional[Array] = None,  # (T, C) or (C,), optional
) -> Array:
    """Compute rate volatility metric by period.

    Formula:
    RV_t = sum_c w_c * | r_{t,c} - r_{t-1,c} |^2

    If ``previous_year_rates`` is None, a one-step lag of ``regulated_rates``
    is used (with RV_0 = 0 by construction).

    Args:
        regulated_rates: Array of shape (T, C) with rates by class.
        class_weights: Array of shape (C,) with weights per class.
        previous_year_rates: Optional array of shape (T, C) or (C,) with prior
            year rates to compare against.

    Returns:
        Array of shape (T,) with rate volatility per period.
    """
    if previous_year_rates is None:
        r_lag = np.concatenate([regulated_rates[:1, :], regulated_rates[:-1, :]], axis=0)
    else:
        r_lag = np.broadcast_to(previous_year_rates, regulated_rates.shape)
    diff = regulated_rates - r_lag
    return (np.square(diff) * class_weights[None, :]).sum(axis=1)


def political_equity_penalty(
    residential_rate: Array,                # shape: (T,)
    regional_benchmark_rate: Array,         # shape: (T,)
    gini_service_index_by_category: Array,  # shape: (T, I)
    kappa_residential_premium: float,       # κ1
    kappa_gini_weight: float,               # κ2
) -> Array:
    """Compute political equity penalty per period.

    Formula:
    PE_t = kappa1 * max(0, r_res,t - r_region,t) + kappa2 * sum_i GINI_{i,t}

    Args:
        residential_rate: Array of shape (T,) with residential rates.
        regional_benchmark_rate: Array of shape (T,) with benchmark regional
            rates.
        gini_service_index_by_category: Array of shape (T, I) with service
            inequality metrics by category.
        kappa_residential_premium: Weight for residential premium over region.
        kappa_gini_weight: Weight for GINI service index contribution.

    Returns:
        Array of shape (T,) with political equity penalty per period.
    """
    premium = np.maximum(0.0, residential_rate - regional_benchmark_rate)
    gini_sum = gini_service_index_by_category.sum(axis=1)
    return kappa_residential_premium * premium + kappa_gini_weight * gini_sum


def economic_development_benefit(
    jobs_by_industry: Array,                # shape: (T, K)
    tax_base_by_industry: Array,            # shape: (T, K)
    industry_rate: Array,                   # shape: (T, K)
    competitive_rate_threshold: Array | float,  # scalar or broadcastable to (T, K)
    weight_jobs: float,                     # ν1
    weight_tax_base: float,                 # ν2
) -> Array:
    """Compute economic development benefit per period.

    Formula:
    ED_t = sum_ind [ Jobs_{t,ind} * v1 + TaxBase_{t,ind} * v2 ] * 1{ r_{t,ind} < r_competitive }

    Args:
        jobs_by_industry: Array of shape (T, K) with job counts by industry.
        tax_base_by_industry: Array of shape (T, K) with tax base by industry.
        industry_rate: Array of shape (T, K) with industrial rates.
        competitive_rate_threshold: Scalar or array broadcastable to (T, K)
            with competitive threshold rates.
        weight_jobs: Weight applied to jobs.
        weight_tax_base: Weight applied to tax base.

    Returns:
        Array of shape (T,) with economic development benefit per period.
    """
    threshold = np.broadcast_to(competitive_rate_threshold, industry_rate.shape)
    indicator = (industry_rate < threshold).astype(float)
    term = weight_jobs * jobs_by_industry + weight_tax_base * tax_base_by_industry
    return (term * indicator).sum(axis=1)


def total_cost_muni(
    operating_expenditure: Array,           # OPEX_t
    power_purchase_cost: Array,             # PowerPurchase_t
    transmission_and_distribution_cost: Array,  # TransDist_t
) -> Array:
    """Compute total municipal costs per period.

    Formula:
    Cost_total,t = OPEX_t + PowerPurchase_t + TransDist_t

    Args:
        operating_expenditure: Array of shape (T,) with OPEX.
        power_purchase_cost: Array of shape (T,) with power purchase cost.
        transmission_and_distribution_cost: Array of shape (T,) with T&D cost.

    Returns:
        Array of shape (T,) with total costs per period.
    """
    return operating_expenditure + power_purchase_cost + transmission_and_distribution_cost


def debt_service_requirement(
    principal_payment: Array,               # Principal_t
    interest_payment: Array,                # Interest_t
    coverage_required: float,               # coverage_required
) -> Array:
    """Compute debt service requirement per period.

    Formula:
    DSR_t = (Principal_t + Interest_t) * coverage_required

    Args:
        principal_payment: Array of shape (T,) with principal payments.
        interest_payment: Array of shape (T,) with interest payments.
        coverage_required: Scalar coverage requirement multiplier.

    Returns:
        Array of shape (T,) with debt service requirement per period.
    """
    return (principal_payment + interest_payment) * coverage_required


def transfer_to_city(
    transfer_rate: float,                   # τ_transfer
    total_revenue: Array,                   # Rev_t = sum_c r_{c,t} * Q_{c,t}
) -> Array:
    """Compute transfer to the city (payment in lieu of taxes) per period.

    Formula:
    Transfer_t = tau_transfer * Rev_t

    Args:
        transfer_rate: Scalar transfer rate.
        total_revenue: Array of shape (T,) with total revenue.

    Returns:
        Array of shape (T,) with transfer amounts per period.
    """
    return transfer_rate * total_revenue


def revenue_sufficiency_slack(
    total_revenue: Array,                   # Rev_t
    total_cost: Array,                      # Cost_total,t
    debt_service_req: Array,                # DSR_t
    transfer_amount: Array,                 # Transfer_t
) -> Array:
    """Compute revenue sufficiency slack per period.

    Formula:
    Slack_t = Rev_t - (Cost_total,t + DSR_t + Transfer_t)

    Constraint satisfied iff Slack_t >= 0.

    Args:
        total_revenue: Array of shape (T,) with total revenue.
        total_cost: Array of shape (T,) with total cost.
        debt_service_req: Array of shape (T,) with debt service requirement.
        transfer_amount: Array of shape (T,) with transfer to city.

    Returns:
        Array of shape (T,) with slack per period.
    """
    return total_revenue - (total_cost + debt_service_req + transfer_amount)


def debt_service_coverage_ratio(
    net_revenue: Array,                     # NetRevenue_t (per muni accounting convention)
    debt_service: Array,                    # DebtService_t (Principal + Interest)
) -> Array:
    """Compute debt service coverage ratio per period.

    Formula:
    DSCR_t = NetRevenue_t / DebtService_t

    Args:
        net_revenue: Array of shape (T,) with net revenue per period.
        debt_service: Array of shape (T,) with debt service per period.

    Returns:
        Array of shape (T,) with DSCR values; inf when debt service is zero.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(debt_service > 0, net_revenue / debt_service, np.inf)


def discount_factors_muni(municipal_discount_rate: float, periods: int) -> Array:
    """Compute discount factors for municipal objective (end-of-period).

    Uses t = 1..T convention:
    df_t = 1 / (1 + r_muni)^t

    Args:
        municipal_discount_rate: Scalar discount rate r_muni.
        periods: Number of periods T.

    Returns:
        Array of shape (T,) with discount factors for t = 1..T.
    """
    return 1.0 / (1.0 + municipal_discount_rate) ** np.arange(1, periods + 1)


@dataclass(frozen=True)
class MunicipalWeights:
    """Weights for municipal objective components.

    Attributes:
        weight_rate: Weight on average rate component (delta1).
        weight_rate_volatility: Weight on rate volatility (delta2).
        weight_equity_penalty: Weight on political equity penalty (delta3).
        weight_economic_development: Weight on economic development benefit
            (delta4).

    Notes:
        Typical usage is to set non-negative values, but bounds are not
        enforced by this class.
    """
    weight_rate: float = 1.0          # δ1
    weight_rate_volatility: float = 1.0  # δ2
    weight_equity_penalty: float = 1.0   # δ3
    weight_economic_development: float = 1.0  # δ4


def municipal_utility_components(
    # Rates & sales
    regulated_rates: Array,                 # (T, C)
    sales_by_customer_class: Array,         # (T, C)
    # Volatility
    class_weights: Array,                   # (C,)
    # Political/equity
    residential_rate: Array,                # (T,)
    regional_benchmark_rate: Array,         # (T,)
    gini_service_index_by_category: Array,  # (T, I)
    # Optional prior-year rates for volatility comparison
    previous_year_rates: Optional[Array] = None,  # (T, C) or (C,), optional
    kappa_residential_premium: float = 1.0,
    kappa_gini_weight: float = 1.0,
    # Economic development
    jobs_by_industry: Array = None,         # (T, K)
    tax_base_by_industry: Array = None,     # (T, K)
    industry_rate: Array = None,            # (T, K)
    competitive_rate_threshold: float | Array = np.inf,
    weight_jobs: float = 1.0,
    weight_tax_base: float = 1.0,
) -> Dict[str, Array]:
    """Compute per-period components for the municipal objective.

    Components computed:
    - RC_t: average rate component
    - RV_t: rate volatility
    - PE_t: political equity penalty
    - ED_t: economic development benefit

    Any of the ED inputs can be omitted (left as None) to yield ED_t = 0.

    Args:
        regulated_rates: (T, C) rates by class.
        sales_by_customer_class: (T, C) sales by class.
        class_weights: (C,) class weights for volatility metric.
        residential_rate: (T,) residential rate.
        regional_benchmark_rate: (T,) regional benchmark rate.
        gini_service_index_by_category: (T, I) service inequality indices.
        previous_year_rates: Optional (T, C) or (C,) prior-year rates to
            compare against for volatility.
        kappa_residential_premium: Weight on residential premium.
        kappa_gini_weight: Weight on GINI service index.
        jobs_by_industry: Optional (T, K) jobs by industry.
        tax_base_by_industry: Optional (T, K) tax base by industry.
        industry_rate: Optional (T, K) industry rates.
        competitive_rate_threshold: Scalar or (T, K) competitive threshold.
        weight_jobs: Weight on jobs in ED term.
        weight_tax_base: Weight on tax base in ED term.

    Returns:
        Dict with keys: "average_rate_component", "rate_volatility",
        "political_equity_penalty", "economic_development_benefit"; each is
        an array of shape (T,).
    """
    rc_t = average_rate_component(regulated_rates, sales_by_customer_class)
    rv_t = rate_volatility(regulated_rates, class_weights, previous_year_rates)
    pe_t = political_equity_penalty(
        residential_rate, regional_benchmark_rate, gini_service_index_by_category,
        kappa_residential_premium, kappa_gini_weight
    )
    if jobs_by_industry is None or tax_base_by_industry is None or industry_rate is None:
        ed_t = np.zeros_like(rc_t)
    else:
        ed_t = economic_development_benefit(
            jobs_by_industry, tax_base_by_industry, industry_rate,
            competitive_rate_threshold, weight_jobs, weight_tax_base
        )
    return {
        "average_rate_component": rc_t,
        "rate_volatility": rv_t,
        "political_equity_penalty": pe_t,
        "economic_development_benefit": ed_t,
    }


def municipal_utility_objective(
    municipal_discount_rate: float,         # r_muni
    regulated_rates: Array,
    sales_by_customer_class: Array,
    class_weights: Array,
    residential_rate: Array,
    regional_benchmark_rate: Array,
    gini_service_index_by_category: Array,
    previous_year_rates: Optional[Array] = None,
    jobs_by_industry: Optional[Array] = None,
    tax_base_by_industry: Optional[Array] = None,
    industry_rate: Optional[Array] = None,
    competitive_rate_threshold: float | Array = np.inf,
    weight_rate: float = 1.0,
    weight_rate_volatility: float = 1.0,
    weight_equity_penalty: float = 1.0,
    weight_economic_development: float = 1.0,
    kappa_residential_premium: float = 1.0,
    kappa_gini_weight: float = 1.0,
    weight_jobs: float = 1.0,
    weight_tax_base: float = 1.0,
) -> float:
    """Compute the municipal utility objective.

    The period objective is a weighted sum of components, discounted and with
    a leading negative sign (per Eq. 26 in the reference):

    W_Muni = - sum_t [ (delta1*RC_t + delta2*RV_t + delta3*PE_t - delta4*ED_t)
                       / (1 + r_muni)^t ]

    Args:
        municipal_discount_rate: Scalar discount rate r_muni.
        regulated_rates: (T, C) rates by class.
        sales_by_customer_class: (T, C) sales by class.
        class_weights: (C,) class weights for RV.
        residential_rate: (T,) residential rate.
        regional_benchmark_rate: (T,) regional benchmark rate.
        gini_service_index_by_category: (T, I) service inequality indices.
        previous_year_rates: Optional (T, C) or (C,) prior-year rates for RV.
        jobs_by_industry: Optional (T, K) jobs by industry.
        tax_base_by_industry: Optional (T, K) tax base by industry.
        industry_rate: Optional (T, K) industry rates.
        competitive_rate_threshold: Scalar or (T, K) competitive threshold.
        weight_rate: Weight on RC_t.
        weight_rate_volatility: Weight on RV_t.
        weight_equity_penalty: Weight on PE_t.
        weight_economic_development: Weight on ED_t (subtracted in objective).
        kappa_residential_premium: Weight on residential premium.
        kappa_gini_weight: Weight on GINI service index.
        weight_jobs: Weight on jobs in ED term.
        weight_tax_base: Weight on tax base in ED term.

    Returns:
        Float objective value (higher is better under this sign convention).
    """
    comps = municipal_utility_components(
        regulated_rates, sales_by_customer_class,
        class_weights,
        residential_rate, regional_benchmark_rate, gini_service_index_by_category,
        previous_year_rates=previous_year_rates,
        kappa_residential_premium=kappa_residential_premium,
        kappa_gini_weight=kappa_gini_weight,
        jobs_by_industry=jobs_by_industry,
        tax_base_by_industry=tax_base_by_industry,
        industry_rate=industry_rate,
        competitive_rate_threshold=competitive_rate_threshold,
        weight_jobs=weight_jobs,
        weight_tax_base=weight_tax_base,
    )

    deltas = MunicipalWeights(
        weight_rate=weight_rate,
        weight_rate_volatility=weight_rate_volatility,
        weight_equity_penalty=weight_equity_penalty,
        weight_economic_development=weight_economic_development,
    )

    # Compose the period objective inside the negative bracket in Eq. (26)
    bracket_t = (
        deltas.weight_rate * comps["average_rate_component"]
        + deltas.weight_rate_volatility * comps["rate_volatility"]
        + deltas.weight_equity_penalty * comps["political_equity_penalty"]
        - deltas.weight_economic_development * comps["economic_development_benefit"]
    )

    T = regulated_rates.shape[0]
    df = discount_factors_muni(municipal_discount_rate, T)

    # W_Muni per Eq. (26): note the leading minus sign.
    return float(-(df * bracket_t).sum())
