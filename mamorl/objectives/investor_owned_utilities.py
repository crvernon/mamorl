from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.floating]


def rate_base(plant_in_service: Array, accumulated_depreciation: Array) -> Array:
    """Compute the per-period rate base.

    The rate base represents net plant in service after accumulated
    depreciation:

    rate_base_t = plant_in_service_t - accumulated_depreciation_t

    Args:
        plant_in_service: Array of shape (T,) for gross plant in service.
        accumulated_depreciation: Array of shape (T,) for accumulated
            depreciation.

    Returns:
        Array of shape (T,) with the rate base for each period.
    """
    return plant_in_service - accumulated_depreciation


def revenue(
    regulated_rates: Array,
    sales_by_customer_class: Array,
    allowed_return_on_rate_base: float,
    rate_base_t: Array,
) -> Array:
    """Compute regulated revenue per period.

    Formula:
    revenue_t = sum_c regulated_rates_{t,c} * sales_{t,c} + r_RB * rate_base_t

    Args:
        regulated_rates: Array of shape (T, C) with rates by period and
            customer class.
        sales_by_customer_class: Array of shape (T, C) with sales volumes by
            period and customer class.
        allowed_return_on_rate_base: Scalar allowed return on rate base
            (r_RB).
        rate_base_t: Array of shape (T,) with the per-period rate base.

    Returns:
        Array of shape (T,) with revenue by period.
    """
    return (regulated_rates * sales_by_customer_class).sum(axis=1) + \
        allowed_return_on_rate_base * rate_base_t


def labor_cost(labor_quantities: Array, wage_rates: Array) -> Array:
    """Compute labor cost per period.

    Formula:
    labor_cost_t = sum_i labor_quantities_{t,i} * wage_rates_i

    Args:
        labor_quantities: Array of shape (T, I) with quantities of labor
            inputs by period and labor type.
        wage_rates: Array of shape (I,) or broadcastable to (T, I) with wage
            rates per labor type.

    Returns:
        Array of shape (T,) with labor cost by period.
    """
    return (labor_quantities * wage_rates).sum(axis=1)


def material_cost(input_quantities: Array, input_prices: Array) -> Array:
    """Compute material cost per period.

    Formula:
    material_cost_t = sum_j input_quantities_{t,j} * input_prices_j

    Args:
        input_quantities: Array of shape (T, J) with material/input
            quantities by period and input type.
        input_prices: Array of shape (J,) or broadcastable to (T, J) with
            prices per input type.

    Returns:
        Array of shape (T,) with material cost by period.
    """
    return (input_quantities * input_prices).sum(axis=1)


def fuel_cost(heat_rate: Array, generation_by_unit: Array, fuel_price: Array) -> Array:
    """Compute fuel cost per period with flexible fuel price shapes.

    Formula:
    fuel_cost_t = sum_g heat_rate_g * generation_{t,g} * fuel_price_{t,g}

    The ``fuel_price`` argument may be provided as a scalar, (G,), (T,), or
    (T, G); it is broadcast to the shape of ``generation_by_unit``.

    Args:
        heat_rate: Array of shape (G,) with heat rate per generating unit.
        generation_by_unit: Array of shape (T, G) with generation by period
            and unit.
        fuel_price: Array broadcastable to shape (T, G) containing fuel
            prices.

    Returns:
        Array of shape (T,) with fuel cost by period.
    """
    fuel_price_tg = np.broadcast_to(fuel_price, generation_by_unit.shape)
    return (heat_rate[None, :] * generation_by_unit * fuel_price_tg).sum(axis=1)


def operating_expenditure(
    labor_cost_t: Array,
    material_cost_t: Array,
    fuel_cost_t: Array,
) -> Array:
    """Compute operating expenditure per period.

    Formula:
    opex_t = labor_cost_t + material_cost_t + fuel_cost_t

    Args:
        labor_cost_t: Array of shape (T,) with labor cost.
        material_cost_t: Array of shape (T,) with material cost.
        fuel_cost_t: Array of shape (T,) with fuel cost.

    Returns:
        Array of shape (T,) with operating expenditure by period.
    """
    return labor_cost_t + material_cost_t + fuel_cost_t


def total_cost(
    operating_expenditure_t: Array,
    capital_expenditures: Array,
    fixed_costs: Array,
    taxes: Array,
) -> Array:
    """Compute total cost per period.

    Formula:
    total_cost_t = opex_t + capex_t + fixed_costs_t + taxes_t

    Args:
        operating_expenditure_t: Array of shape (T,) with operating
            expenditure.
        capital_expenditures: Array of shape (T,) with capital expenditures.
        fixed_costs: Array of shape (T,) with fixed costs.
        taxes: Array of shape (T,) with taxes.

    Returns:
        Array of shape (T,) with total cost by period.
    """
    return operating_expenditure_t + capital_expenditures + fixed_costs + taxes


def performance_based_rate_adjustment(
    performance_weights: Array,
    performance_metrics: Array,
    performance_targets: Array,
) -> Array:
    """Compute performance-based rate adjustment (PBR) per period.

    Formula:
    pbr_t = sum_m w_m * (metric_{t,m} - target_{t,m})

    Args:
        performance_weights: Array of shape (M,) with weights per metric.
        performance_metrics: Array of shape (T, M) with realized metrics.
        performance_targets: Array of shape (T, M) with target metrics.

    Returns:
        Array of shape (T,) with the PBR term per period.
    """
    return (performance_weights * (performance_metrics - performance_targets)).sum(axis=1)


def reliability_penalties(
    saidi_penalty_weight: float,
    saifi_penalty_weight: float,
    system_average_interruption_duration_index: Array,
    saidi_target: float,
    system_average_interruption_frequency_index: Array,
    saifi_target: float,
) -> Array:
    """Compute reliability penalty per period.

    Formula:
    rel_t = lambda1 * max(0, SAIDI_t - SAIDI*)
            + lambda2 * max(0, SAIFI_t - SAIFI*)

    Args:
        saidi_penalty_weight: Weight applied to SAIDI deviations.
        saifi_penalty_weight: Weight applied to SAIFI deviations.
        system_average_interruption_duration_index: Array of shape (T,)
            with SAIDI values.
        saidi_target: Target SAIDI value.
        system_average_interruption_frequency_index: Array of shape (T,)
            with SAIFI values.
        saifi_target: Target SAIFI value.

    Returns:
        Array of shape (T,) with reliability penalties by period.
    """
    saidi_term = np.maximum(0.0, system_average_interruption_duration_index - saidi_target)
    saifi_term = np.maximum(0.0, system_average_interruption_frequency_index - saifi_target)
    return saidi_penalty_weight * saidi_term + saifi_penalty_weight * saifi_term


def environmental_social_governance_term(
    esg_weight_rps: float,
    esg_weight_co2_reduction: float,
    esg_weight_emissions: float,
    renewable_portfolio_share: Array,
    carbon_dioxide_reduction: Array,
    emissions: Array,
) -> Array:
    """Compute the ESG contribution per period.

    Positive terms for renewable portfolio share and CO2 reduction, and a
    negative term for emissions:

    esg_t = w1 * RPS_t + w2 * CO2Red_t - w3 * Emissions_t

    Args:
        esg_weight_rps: Weight on renewable portfolio share.
        esg_weight_co2_reduction: Weight on carbon dioxide reduction.
        esg_weight_emissions: Weight on emissions (penalty).
        renewable_portfolio_share: Array of shape (T,) with RPS values.
        carbon_dioxide_reduction: Array of shape (T,) with CO2 reduction.
        emissions: Array of shape (T,) with emissions.

    Returns:
        Array of shape (T,) with ESG terms by period.
    """
    return (
        esg_weight_rps * renewable_portfolio_share
        + esg_weight_co2_reduction * carbon_dioxide_reduction
        - esg_weight_emissions * emissions
    )


def discount_factors(discount_rate: float, periods: int) -> Array:
    """Compute per-period discount factors.

    Uses end-of-period discounting for t = 1..periods:
    df_t = 1 / (1 + r)^t

    Args:
        discount_rate: Scalar discount rate r.
        periods: Number of periods to compute.

    Returns:
        Array of shape (periods,) with discount factors for t = 1..periods.
    """
    return 1.0 / (1.0 + discount_rate) ** np.arange(1, periods + 1)


@dataclass(frozen=True)
class ObjectiveWeights:
    """Container for scalar weights on objective components.

    Attributes:
        weight_pbr: Weight applied to performance-based rate adjustment.
        weight_rel: Weight applied to reliability penalties.
        weight_esg: Weight applied to ESG term.

    Notes:
        Typical usage is to set non-negative values, but this class does not
        enforce bounds.
    """
    weight_pbr: float = 1.0
    weight_rel: float = 1.0
    weight_esg: float = 1.0


def cash_flow(
    revenue_t: Array,
    total_cost_t: Array,
    pbr_t: Array,
    rel_t: Array,
    esg_t: Array,
    weights: ObjectiveWeights,
) -> Array:
    """Compute cash flow per period combining revenue, costs and modifiers.

    Formula:
    cash_flow_t = revenue_t - total_cost_t
                  + w_pbr * pbr_t - w_rel * rel_t + w_esg * esg_t

    Args:
        revenue_t: Array of shape (T,) with revenue.
        total_cost_t: Array of shape (T,) with total cost.
        pbr_t: Array of shape (T,) with performance-based rate adjustment.
        rel_t: Array of shape (T,) with reliability penalties.
        esg_t: Array of shape (T,) with ESG term.
        weights: ``ObjectiveWeights`` instance with scalar weights.

    Returns:
        Array of shape (T,) with cash flow by period.
    """
    return (
        revenue_t
        - total_cost_t
        + weights.weight_pbr * pbr_t
        - weights.weight_rel * rel_t
        + weights.weight_esg * esg_t
    )


def investor_owned_utility_components(
    regulated_rates: Array,
    sales_by_customer_class: Array,
    allowed_return_on_rate_base: float,
    plant_in_service: Array,
    accumulated_depreciation: Array,
    wage_rates: Array,
    labor_quantities: Array,
    input_prices: Array,
    input_quantities: Array,
    heat_rate: Array,
    generation_by_unit: Array,
    fuel_price: Array,
    capital_expenditures: Array,
    fixed_costs: Array,
    taxes: Array,
    performance_weights: Array,
    performance_metrics: Array,
    performance_targets: Array,
    saidi_penalty_weight: float,
    saifi_penalty_weight: float,
    system_average_interruption_duration_index: Array,
    saidi_target: float,
    system_average_interruption_frequency_index: Array,
    saifi_target: float,
    esg_weight_rps: float,
    esg_weight_co2_reduction: float,
    esg_weight_emissions: float,
    renewable_portfolio_share: Array,
    carbon_dioxide_reduction: Array,
    emissions: Array,
) -> Dict[str, Array]:
    """Compute and return per-period components for the IOU objective.

    This is a convenience function that computes each intermediate time
    series used by the investor-owned utility (IOU) objective, including
    revenue, costs, PBR, reliability penalties, and ESG terms.

    Args:
        regulated_rates: (T, C) rates by class.
        sales_by_customer_class: (T, C) sales by class.
        allowed_return_on_rate_base: Scalar return on rate base.
        plant_in_service: (T,) gross plant in service.
        accumulated_depreciation: (T,) accumulated depreciation.
        wage_rates: (I,) or broadcastable to (T, I) wage rates.
        labor_quantities: (T, I) labor quantities.
        input_prices: (J,) or broadcastable to (T, J) input prices.
        input_quantities: (T, J) input quantities.
        heat_rate: (G,) heat rate by unit.
        generation_by_unit: (T, G) generation by unit.
        fuel_price: Broadcastable to (T, G) fuel prices.
        capital_expenditures: (T,) capex.
        fixed_costs: (T,) fixed costs.
        taxes: (T,) taxes.
        performance_weights: (M,) PBR weights.
        performance_metrics: (T, M) realized metrics.
        performance_targets: (T, M) targets.
        saidi_penalty_weight: SAIDI penalty weight.
        saifi_penalty_weight: SAIFI penalty weight.
        system_average_interruption_duration_index: (T,) SAIDI values.
        saidi_target: Target SAIDI.
        system_average_interruption_frequency_index: (T,) SAIFI values.
        saifi_target: Target SAIFI.
        esg_weight_rps: Weight on RPS.
        esg_weight_co2_reduction: Weight on CO2 reduction.
        esg_weight_emissions: Weight on emissions.
        renewable_portfolio_share: (T,) RPS.
        carbon_dioxide_reduction: (T,) CO2 reduction.
        emissions: (T,) emissions.

    Returns:
        Dict mapping component names to arrays of shape (T,):
        {"rate_base", "revenue", "labor_cost", "material_cost",
        "fuel_cost", "operating_expenditure", "total_cost",
        "performance_based_rate_adjustment", "reliability_penalties",
        "environmental_social_governance_term"}.
    """
    rb_t = rate_base(plant_in_service, accumulated_depreciation)
    revenue_t = revenue(
        regulated_rates, sales_by_customer_class, allowed_return_on_rate_base, rb_t
    )

    labor_cost_t = labor_cost(labor_quantities, wage_rates)
    material_cost_t = material_cost(input_quantities, input_prices)
    fuel_cost_t = fuel_cost(heat_rate, generation_by_unit, fuel_price)
    opex_t = operating_expenditure(labor_cost_t, material_cost_t, fuel_cost_t)

    total_cost_t = total_cost(opex_t, capital_expenditures, fixed_costs, taxes)
    pbr_t = performance_based_rate_adjustment(
        performance_weights, performance_metrics, performance_targets
    )
    rel_t = reliability_penalties(
        saidi_penalty_weight,
        saifi_penalty_weight,
        system_average_interruption_duration_index,
        saidi_target,
        system_average_interruption_frequency_index,
        saifi_target,
    )
    esg_t = environmental_social_governance_term(
        esg_weight_rps,
        esg_weight_co2_reduction,
        esg_weight_emissions,
        renewable_portfolio_share,
        carbon_dioxide_reduction,
        emissions,
    )

    return {
        "rate_base": rb_t,
        "revenue": revenue_t,
        "labor_cost": labor_cost_t,
        "material_cost": material_cost_t,
        "fuel_cost": fuel_cost_t,
        "operating_expenditure": opex_t,
        "total_cost": total_cost_t,
        "performance_based_rate_adjustment": pbr_t,
        "reliability_penalties": rel_t,
        "environmental_social_governance_term": esg_t,
    }


def investor_owned_utility_objective(
    discount_rate: float,
    regulated_rates: Array,
    sales_by_customer_class: Array,
    allowed_return_on_rate_base: float,
    plant_in_service: Array,
    accumulated_depreciation: Array,
    wage_rates: Array,
    labor_quantities: Array,
    input_prices: Array,
    input_quantities: Array,
    heat_rate: Array,
    generation_by_unit: Array,
    fuel_price: Array,
    capital_expenditures: Array,
    fixed_costs: Array,
    taxes: Array,
    performance_weights: Array,
    performance_metrics: Array,
    performance_targets: Array,
    saidi_penalty_weight: float,
    saifi_penalty_weight: float,
    system_average_interruption_duration_index: Array,
    saidi_target: float,
    system_average_interruption_frequency_index: Array,
    saifi_target: float,
    esg_weight_rps: float,
    esg_weight_co2_reduction: float,
    esg_weight_emissions: float,
    renewable_portfolio_share: Array,
    carbon_dioxide_reduction: Array,
    emissions: Array,
    weight_pbr: float = 1.0,
    weight_rel: float = 1.0,
    weight_esg: float = 1.0,
) -> float:
    """Compute the net present value (NPV) of the IOU objective.

    This function aggregates revenue, costs, performance-based rate
    adjustments, reliability penalties, and ESG terms into a cash flow
    series, applies user-specified weights, discounts it, and returns the
    NPV.

    Args:
        discount_rate: Scalar discount rate used to compute discount factors.
        regulated_rates: (T, C) rates by class.
        sales_by_customer_class: (T, C) sales by class.
        allowed_return_on_rate_base: Scalar return on rate base.
        plant_in_service: (T,) gross plant in service.
        accumulated_depreciation: (T,) accumulated depreciation.
        wage_rates: (I,) or broadcastable to (T, I) wage rates.
        labor_quantities: (T, I) labor quantities.
        input_prices: (J,) or broadcastable to (T, J) input prices.
        input_quantities: (T, J) input quantities.
        heat_rate: (G,) heat rate by unit.
        generation_by_unit: (T, G) generation by unit.
        fuel_price: Broadcastable to (T, G) fuel prices.
        capital_expenditures: (T,) capex.
        fixed_costs: (T,) fixed costs.
        taxes: (T,) taxes.
        performance_weights: (M,) PBR weights.
        performance_metrics: (T, M) realized metrics.
        performance_targets: (T, M) targets.
        saidi_penalty_weight: SAIDI penalty weight.
        saifi_penalty_weight: SAIFI penalty weight.
        system_average_interruption_duration_index: (T,) SAIDI values.
        saidi_target: Target SAIDI.
        system_average_interruption_frequency_index: (T,) SAIFI values.
        saifi_target: Target SAIFI.
        esg_weight_rps: Weight on RPS.
        esg_weight_co2_reduction: Weight on CO2 reduction.
        esg_weight_emissions: Weight on emissions.
        renewable_portfolio_share: (T,) RPS.
        carbon_dioxide_reduction: (T,) CO2 reduction.
        emissions: (T,) emissions.
        weight_pbr: Weight applied to PBR term in cash flow.
        weight_rel: Weight applied to reliability penalty in cash flow.
        weight_esg: Weight applied to ESG term in cash flow.

    Returns:
        Float NPV of the IOU objective.
    """
    components = investor_owned_utility_components(
        regulated_rates,
        sales_by_customer_class,
        allowed_return_on_rate_base,
        plant_in_service,
        accumulated_depreciation,
        wage_rates,
        labor_quantities,
        input_prices,
        input_quantities,
        heat_rate,
        generation_by_unit,
        fuel_price,
        capital_expenditures,
        fixed_costs,
        taxes,
        performance_weights,
        performance_metrics,
        performance_targets,
        saidi_penalty_weight,
        saifi_penalty_weight,
        system_average_interruption_duration_index,
        saidi_target,
        system_average_interruption_frequency_index,
        saifi_target,
        esg_weight_rps,
        esg_weight_co2_reduction,
        esg_weight_emissions,
        renewable_portfolio_share,
        carbon_dioxide_reduction,
        emissions,
    )

    weights = ObjectiveWeights(
        weight_pbr=weight_pbr, weight_rel=weight_rel, weight_esg=weight_esg
    )
    cf_t = cash_flow(
        components["revenue"],
        components["total_cost"],
        components["performance_based_rate_adjustment"],
        components["reliability_penalties"],
        components["environmental_social_governance_term"],
        weights,
    )

    periods = regulated_rates.shape[0]
    df_t = discount_factors(discount_rate, periods)
    return float((df_t * cf_t).sum())
