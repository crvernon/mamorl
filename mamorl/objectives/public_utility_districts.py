from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.floating]


def consumer_surplus_approx(
    price_elasticity_by_class: Array,      # (C,)
    sales_by_customer_class: Array,        # (T, C) -> Q_{c,t}
    price_cap_by_class: Array,             # (C,) -> P_{c,max}
    regulated_rates: Array,                # (T, C) -> r_{c,t}
) -> Array:
    """Approximate consumer surplus per period by customer class.

    Formula:
    CS_t ≈ sum_c [ 0.5 * eps_c * Q_{t,c} * (P_{max,c} - r_{t,c})^2 / r_{t,c} ]

    Args:
        price_elasticity_by_class: Array of shape (C,) with price elasticities
            by class (eps_c).
        sales_by_customer_class: Array of shape (T, C) with sales quantities
            by period and class (Q_{t,c}).
        price_cap_by_class: Array of shape (C,) with price caps (P_{max,c}).
        regulated_rates: Array of shape (T, C) with regulated rates by period
            and class (r_{t,c}).

    Returns:
        Array of shape (T,) with consumer surplus approximation per period.
    """
    q_tc = sales_by_customer_class
    r_tc = regulated_rates
    pmax_c = price_cap_by_class[None, :]
    eps_c = price_elasticity_by_class[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        term = 0.5 * eps_c * q_tc * np.square(pmax_c - r_tc) / r_tc
        term = np.where(r_tc > 0, term, 0.0)
    return term.sum(axis=1)  # (T,)


def producer_surplus(
    wholesale_price: Array,                # (T,) or (T, G) -> P_wholesale,t
    marginal_cost_by_unit: Array,          # (G,) or (T, G) -> MC_{g,t}
    generation_by_unit: Array,             # (T, G) -> Gen_{g,t}
    use_positive_surplus_only: bool = True,
) -> Array:
    """Compute producer surplus per period.

    Formula:
    PS_t = sum_g (P_{t,g} - MC_{t,g}) * Gen_{t,g} * 1{surplus}

    Where 1{surplus} optionally clips negative margins to zero when
    ``use_positive_surplus_only`` is True.

    Broadcasting rules:
    - ``wholesale_price``: (T,) is broadcast across units to (T, G)
    - ``marginal_cost_by_unit``: (G,) is broadcast across periods to (T, G)

    Args:
        wholesale_price: Array of shape (T,) or (T, G) with wholesale prices.
        marginal_cost_by_unit: Array of shape (G,) or (T, G) with marginal
            costs.
        generation_by_unit: Array of shape (T, G) with generation by unit.
        use_positive_surplus_only: If True, negative margins are set to zero.

    Returns:
        Array of shape (T,) with producer surplus per period.
    """
    p_tg = np.broadcast_to(wholesale_price if wholesale_price.ndim == 2
                           else wholesale_price[:, None],
                           generation_by_unit.shape)
    mc_tg = np.broadcast_to(marginal_cost_by_unit, generation_by_unit.shape)
    margin_tg = p_tg - mc_tg
    if use_positive_surplus_only:
        margin_tg = np.maximum(margin_tg, 0.0)
    return (margin_tg * generation_by_unit).sum(axis=1)  # (T,)


def external_costs(
    social_cost_of_carbon: float,          # φ_CO2
    emissions: Array,                      # (T,)
    local_externality_cost: float,         # φ_local
    local_pollution_index: Array,          # (T,)
) -> Array:
    """Compute external costs per period from emissions and local pollution.

    Formula:
    ExtCost_t = phi_CO2 * Emissions_t + phi_local * LocalPollution_t

    Args:
        social_cost_of_carbon: Scalar social cost per unit of emissions.
        emissions: Array of shape (T,) with emissions per period.
        local_externality_cost: Scalar cost per unit of local pollution.
        local_pollution_index: Array of shape (T,) with local pollution index.

    Returns:
        Array of shape (T,) with external costs per period.
    """
    return social_cost_of_carbon * emissions + local_externality_cost * local_pollution_index


def local_benefits(
    employment_weights: Array,             # (J,) -> w_j
    local_employment_by_category: Array,   # (T, J)
    purchase_weights: Array,               # (K,) -> ψ_k
    local_purchases_by_category: Array,    # (T, K)
) -> Array:
    """Compute local benefits per period from employment and purchases.

    Formula:
    LocalBenefit_t = sum_j w_j * LocalEmployment_{t,j}
                     + sum_k psi_k * LocalPurchase_{t,k}

    Args:
        employment_weights: Array of shape (J,) with job weights w_j.
        local_employment_by_category: Array of shape (T, J) with local
            employment by category.
        purchase_weights: Array of shape (K,) with purchase weights psi_k.
        local_purchases_by_category: Array of shape (T, K) with local
            purchases by category.

    Returns:
        Array of shape (T,) with local benefits per period.
    """
    jobs = (local_employment_by_category * employment_weights[None, :]).sum(axis=1)
    purchases = (local_purchases_by_category * purchase_weights[None, :]).sum(axis=1)
    return jobs + purchases


def total_operational_cost_pud(
    generation_cost: Array,                # (T,)
    power_purchase_cost: Array,            # (T,)
    transmission_and_distribution_cost: Array,  # (T,)
) -> Array:
    """Compute total operational costs for a PUD per period.

    Formula:
    TC_PUD,t = GenCost_t + PurchaseCost_t + T&D_t

    Args:
        generation_cost: Array of shape (T,) with generation cost.
        power_purchase_cost: Array of shape (T,) with power purchase cost.
        transmission_and_distribution_cost: Array of shape (T,) with T&D cost.

    Returns:
        Array of shape (T,) with total operational costs per period.
    """
    return generation_cost + power_purchase_cost + transmission_and_distribution_cost


def discount_factors_social(social_discount_rate: float, periods: int) -> Array:
    """Compute social discount factors for end-of-period cash flows.

    Uses t = 1..T convention:
    df_t = 1 / (1 + r_social)^t

    Args:
        social_discount_rate: Scalar social discount rate r_social.
        periods: Number of periods T.

    Returns:
        Array of shape (T,) with discount factors for t = 1..T.
    """
    return 1.0 / (1.0 + social_discount_rate) ** np.arange(1, periods + 1)


@dataclass(frozen=True)
class PudComponents:
    """Container for PUD welfare components by period.

    Attributes:
        consumer_surplus: Array of shape (T,) with consumer surplus CS_t.
        producer_surplus: Array of shape (T,) with producer surplus PS_t.
        external_costs: Array of shape (T,) with external costs.
        local_benefits: Array of shape (T,) with local benefits.
    """
    consumer_surplus: Array
    producer_surplus: Array
    external_costs: Array
    local_benefits: Array

    @property
    def period_welfare(self) -> Array:
        """Total welfare per period combining all components.

        Formula:
        W_t = CS_t + PS_t - ExtCost_t + LocalBenefit_t

        Returns:
            Array of shape (T,) with welfare per period.
        """
        return self.consumer_surplus + self.producer_surplus - self.external_costs + self.local_benefits


def public_utility_district_components(
    # Consumer surplus approximation inputs
    price_elasticity_by_class: Array,      # (C,)
    sales_by_customer_class: Array,        # (T, C)
    price_cap_by_class: Array,             # (C,)
    regulated_rates: Array,                # (T, C)
    # Producer surplus inputs
    wholesale_price: Array,                # (T,) or (T, G)
    marginal_cost_by_unit: Array,          # (G,) or (T, G)
    generation_by_unit: Array,             # (T, G)
    # External costs
    social_cost_of_carbon: float,
    emissions: Array,                      # (T,)
    local_externality_cost: float,
    local_pollution_index: Array,          # (T,)
    # Local benefits
    employment_weights: Array,             # (J,)
    local_employment_by_category: Array,   # (T, J)
    purchase_weights: Array,               # (K,)
    local_purchases_by_category: Array,    # (T, K)
) -> PudComponents:
    """Compute per-period components for the PUD welfare objective.

    Returns the time series needed to evaluate WPUD (Eq. 40): consumer
    surplus, producer surplus, external costs, and local benefits.

    Args:
        price_elasticity_by_class: (C,) elasticities by class.
        sales_by_customer_class: (T, C) sales by class.
        price_cap_by_class: (C,) price cap by class.
        regulated_rates: (T, C) regulated rates by class.
        wholesale_price: (T,) or (T, G) wholesale prices.
        marginal_cost_by_unit: (G,) or (T, G) marginal costs by unit.
        generation_by_unit: (T, G) generation by unit.
        social_cost_of_carbon: Scalar social cost per unit of emissions.
        emissions: (T,) emissions per period.
        local_externality_cost: Scalar local externality cost.
        local_pollution_index: (T,) local pollution index.
        employment_weights: (J,) employment weights.
        local_employment_by_category: (T, J) local employment.
        purchase_weights: (K,) purchase weights.
        local_purchases_by_category: (T, K) local purchases.

    Returns:
        ``PudComponents`` containing arrays of shape (T,) for each component.
    """
    cs_t = consumer_surplus_approx(
        price_elasticity_by_class, sales_by_customer_class, price_cap_by_class, regulated_rates
    )
    ps_t = producer_surplus(wholesale_price, marginal_cost_by_unit, generation_by_unit)
    ext_t = external_costs(social_cost_of_carbon, emissions, local_externality_cost, local_pollution_index)
    loc_t = local_benefits(
        employment_weights, local_employment_by_category, purchase_weights, local_purchases_by_category
    )
    return PudComponents(
        consumer_surplus=cs_t,
        producer_surplus=ps_t,
        external_costs=ext_t,
        local_benefits=loc_t,
    )


def public_utility_district_objective(
    social_discount_rate: float,           # r_social
    price_elasticity_by_class: Array,
    sales_by_customer_class: Array,
    price_cap_by_class: Array,
    regulated_rates: Array,
    wholesale_price: Array,
    marginal_cost_by_unit: Array,
    generation_by_unit: Array,
    social_cost_of_carbon: float,
    emissions: Array,
    local_externality_cost: float,
    local_pollution_index: Array,
    employment_weights: Array,
    local_employment_by_category: Array,
    purchase_weights: Array,
    local_purchases_by_category: Array,
) -> float:
    """Compute the PUD welfare objective (WPUD).

    Formula:
    WPUD = sum_t [ (CS_t + PS_t - ExtCost_t + LocalBenefit_t) / (1 + r_social)^t ]

    Args:
        social_discount_rate: Scalar social discount rate r_social.
        price_elasticity_by_class: (C,) price elasticities by class.
        sales_by_customer_class: (T, C) sales by class.
        price_cap_by_class: (C,) price cap by class.
        regulated_rates: (T, C) regulated rates by class.
        wholesale_price: (T,) or (T, G) wholesale prices.
        marginal_cost_by_unit: (G,) or (T, G) marginal costs.
        generation_by_unit: (T, G) generation by unit.
        social_cost_of_carbon: Scalar social cost per unit of emissions.
        emissions: (T,) emissions per period.
        local_externality_cost: Scalar local externality cost.
        local_pollution_index: (T,) local pollution index.
        employment_weights: (J,) employment weights.
        local_employment_by_category: (T, J) local employment.
        purchase_weights: (K,) purchase weights.
        local_purchases_by_category: (T, K) local purchases.

    Returns:
        Float welfare objective value.
    """
    comps = public_utility_district_components(
        price_elasticity_by_class, sales_by_customer_class, price_cap_by_class, regulated_rates,
        wholesale_price, marginal_cost_by_unit, generation_by_unit,
        social_cost_of_carbon, emissions, local_externality_cost, local_pollution_index,
        employment_weights, local_employment_by_category, purchase_weights, local_purchases_by_category,
    )
    T = sales_by_customer_class.shape[0]
    df = discount_factors_social(social_discount_rate, T)
    return float((df * comps.period_welfare).sum())


def equity_rate_ratio(
    rural_residential_rate: Array,          # (T,)
    urban_residential_rate: Array,          # (T,)
) -> Array:
    """Compute the equity rate ratio per period.

    Formula:
    ratio_t = r_rural,t / r_urban,t

    Args:
        rural_residential_rate: Array of shape (T,) with rural residential rate.
        urban_residential_rate: Array of shape (T,) with urban residential rate.

    Returns:
        Array of shape (T,) with ratios; inf when urban rate is zero.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(urban_residential_rate > 0,
                        rural_residential_rate / urban_residential_rate,
                        np.inf)


def equity_constraint_satisfied(
    rural_residential_rate: Array,
    urban_residential_rate: Array,
    equity_tolerance: float,                # ε_equity
) -> Array:
    """Check whether the equity rate constraint is satisfied.

    Condition (Eq. 47):
    r_rural,t / r_urban,t ≤ 1 + epsilon_equity

    Args:
        rural_residential_rate: Array of shape (T,) rural residential rate.
        urban_residential_rate: Array of shape (T,) urban residential rate.
        equity_tolerance: Scalar tolerance epsilon_equity.

    Returns:
        Boolean array of shape (T,) indicating satisfaction per period.
    """
    return equity_rate_ratio(rural_residential_rate, urban_residential_rate) <= (1.0 + equity_tolerance)


def service_quality_constraint(
    service_quality_by_district: Array,     # (T, I_districts)
    minimum_service_quality: float,         # SQ
) -> Array:
    """Check service quality meets or exceeds the minimum threshold.

    Condition (Eq. 48):
    ServiceQuality_{district,i} ≥ SQ

    Args:
        service_quality_by_district: Array of shape (T, I) with service
            quality metrics per district.
        minimum_service_quality: Scalar SQ threshold.

    Returns:
        Boolean array of shape (T, I) indicating satisfaction.
    """
    return service_quality_by_district >= minimum_service_quality


def revenue_equals_cost_slack(
    total_revenue: Array,                   # (T,) Σ_c r_{c,t} Q_{c,t}
    total_cost: Array,                      # (T,) TotalCost_t
) -> Array:
    """Compute slack for revenue equals cost condition (Eq. 49).

    Formula:
    slack_t = Rev_t - TotalCost_t

    Args:
        total_revenue: Array of shape (T,) with total revenue.
        total_cost: Array of shape (T,) with total cost.

    Returns:
        Array of shape (T,) with slack; zero indicates exact satisfaction.
    """
    return total_revenue - total_cost


def operating_reserve_requirement_met(
    operating_reserves: Array,              # (T,)
    annual_operating_expenditure: Array,    # (T,)
    required_fraction: float = 0.15,        # Eq. 50
) -> Array:
    """Check operating reserve requirement (Eq. 50) is satisfied.

    Condition:
    OpReserve_t ≥ required_fraction · AnnualOpEx_t

    Args:
        operating_reserves: Array of shape (T,) with operating reserves.
        annual_operating_expenditure: Array of shape (T,) with annual OPEX.
        required_fraction: Scalar fraction (default 0.15).

    Returns:
        Boolean array of shape (T,) indicating satisfaction per period.
    """
    return operating_reserves >= required_fraction * annual_operating_expenditure


def capital_reserve_requirement_met(
    capital_reserves: Array,                # (T,)
    plant_value: Array,                     # (T,)
    required_fraction: float = 0.10,        # Eq. 51
) -> Array:
    """Check capital reserve requirement (Eq. 51) is satisfied.

    Condition:
    CapReserve_t ≥ required_fraction · PlantValue_t

    Args:
        capital_reserves: Array of shape (T,) with capital reserves.
        plant_value: Array of shape (T,) with plant value.
        required_fraction: Scalar fraction (default 0.10).

    Returns:
        Boolean array of shape (T,) indicating satisfaction per period.
    """
    return capital_reserves >= required_fraction * plant_value


def regional_preference_ratio(
    local_generation: Array,                # (T,)
    local_purchases: Array,                 # (T,)
    total_supply: Array,                    # (T,)
) -> Array:
    """Compute regional preference ratio per period (Eq. 52).

    Formula:
    ratio_t = (LocalGeneration_t + LocalPurchases_t) / TotalSupply_t

    Args:
        local_generation: Array of shape (T,) with local generation.
        local_purchases: Array of shape (T,) with local purchases.
        total_supply: Array of shape (T,) with total supply.

    Returns:
        Array of shape (T,) with ratio; zero when total supply is zero.
    """
    numerator = local_generation + local_purchases
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total_supply > 0, numerator / total_supply, 0.0)


def regional_preference_constraint_met(
    local_generation: Array,
    local_purchases: Array,
    total_supply: Array,
    theta_local: float,                     # θ_local
) -> Array:
    """Check that regional preference ratio meets the minimum threshold.

    Condition (Eq. 52):
    (LocalGeneration_t + LocalPurchases_t) / TotalSupply_t ≥ theta_local

    Args:
        local_generation: Array of shape (T,) with local generation.
        local_purchases: Array of shape (T,) with local purchases.
        total_supply: Array of shape (T,) with total supply.
        theta_local: Scalar minimum regional preference threshold.

    Returns:
        Boolean array of shape (T,) indicating satisfaction per period.
    """
    return regional_preference_ratio(local_generation, local_purchases, total_supply) >= theta_local
