from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.floating]


def energy_revenue(
    spot_price_by_hour: Array,              # shape: (T, H)
    generation_merchant_by_hour: Array,     # shape: (T, H)
    power_purchase_agreement_price: Array,  # shape: (K,) or (T, K)
    generation_contracted: Array,           # shape: (T, K)
) -> Array:
    """Compute energy market revenue per period.

    Formula:
    Rev_energy_t = sum_h P_spot[t,h] * Gen_merchant[t,h]
                   + sum_k P_PPA[(t),k] * Gen_contracted[t,k]

    ``power_purchase_agreement_price`` may be provided as (K,) or (T, K) and
    will be broadcast to the shape of ``generation_contracted`` as needed.

    Args:
        spot_price_by_hour: Array of shape (T, H) with hourly spot prices by
            period.
        generation_merchant_by_hour: Array of shape (T, H) with hourly
            merchant generation.
        power_purchase_agreement_price: Array of shape (K,) or (T, K) with
            PPA prices per contract.
        generation_contracted: Array of shape (T, K) with contracted energy
            quantities per contract.

    Returns:
        Array of shape (T,) with energy revenue per period.
    """
    rev_merchant = (spot_price_by_hour * generation_merchant_by_hour).sum(axis=1)
    ppa_price_tk = np.broadcast_to(power_purchase_agreement_price, generation_contracted.shape)
    rev_ppa = (ppa_price_tk * generation_contracted).sum(axis=1)
    return rev_merchant + rev_ppa


def capacity_revenue(
    capacity_available: Array,   # (T,)
    capacity_price: Array,       # (T,)
    availability_factor: Array,  # (T,)
) -> Array:
    """Compute capacity market revenue per period.

    Formula:
    Rev_capacity_t = Cap_available_t * P_cap_t * AF_t

    Args:
        capacity_available: Array of shape (T,) with available capacity.
        capacity_price: Array of shape (T,) with capacity prices.
        availability_factor: Array of shape (T,) with availability factors.

    Returns:
        Array of shape (T,) with capacity revenue per period.
    """
    return capacity_available * capacity_price * availability_factor


def ancillary_services_revenue(
    ancillary_service_price: Array,     # (T, S)
    ancillary_service_quantity: Array,  # (T, S)
) -> Array:
    """Compute ancillary services revenue per period.

    Formula:
    Rev_AS_t = sum_s P_AS[t,s] * Q_AS[t,s]

    Args:
        ancillary_service_price: Array of shape (T, S) with prices by service.
        ancillary_service_quantity: Array of shape (T, S) with quantities by
            service.

    Returns:
        Array of shape (T,) with ancillary services revenue per period.
    """
    return (ancillary_service_price * ancillary_service_quantity).sum(axis=1)


def fuel_cost_ipps(
    heat_rate: Array,                     # scalar or (T, H) or broadcastable to generation shape
    generation_by_hour: Array,            # (T, H)
    fuel_price_by_hour: Array,            # (T, H) or broadcastable
) -> Array:
    """Compute fuel cost per period for hourly generation.

    Formula:
    FuelCosts_t = sum_h HR[(t),h] * Gen[t,h] * P_fuel[t,h]

    ``heat_rate`` and ``fuel_price_by_hour`` may be provided as scalars or
    arrays broadcastable to the shape of ``generation_by_hour``; broadcasting
    will be applied as needed.

    Args:
        heat_rate: Scalar or array broadcastable to (T, H) with heat rate.
        generation_by_hour: Array of shape (T, H) with hourly generation.
        fuel_price_by_hour: Scalar or array broadcastable to (T, H) with fuel
            price.

    Returns:
        Array of shape (T,) with fuel costs per period.
    """
    hr = np.broadcast_to(heat_rate, generation_by_hour.shape)
    pf = np.broadcast_to(fuel_price_by_hour, generation_by_hour.shape)
    return (hr * generation_by_hour * pf).sum(axis=1)


def variable_om_cost(
    variable_om_rate: Array,              # scalar or (T,)
    generation_by_hour: Array,            # (T, H)
) -> Array:
    """Compute variable operations and maintenance cost per period.

    Formula:
    VOM_t = VOM_rate_t * sum_h Gen[t,h]

    ``variable_om_rate`` may be a scalar or (T,). It is broadcast to (T,) as
    necessary.

    Args:
        variable_om_rate: Scalar or array of shape (T,) with variable O&M
            rate.
        generation_by_hour: Array of shape (T, H) with hourly generation.

    Returns:
        Array of shape (T,) with variable O&M cost per period.
    """
    vom = np.broadcast_to(variable_om_rate, (generation_by_hour.shape[0],))
    energy = generation_by_hour.sum(axis=1)
    return vom * energy


def fixed_om_cost(
    fixed_om_rate: Array,                 # scalar or (T,)
    capacity_installed: Array,            # (T,)
) -> Array:
    """Compute fixed operations and maintenance cost per period.

    Formula:
    FOM_t = FOM_rate_t * Cap_installed_t

    Args:
        fixed_om_rate: Scalar or array of shape (T,) with fixed O&M rate.
        capacity_installed: Array of shape (T,) with installed capacity.

    Returns:
        Array of shape (T,) with fixed O&M cost per period.
    """
    fom = np.broadcast_to(fixed_om_rate, capacity_installed.shape)
    return fom * capacity_installed


def operating_costs(
    heat_rate: Array,
    generation_by_hour: Array,
    fuel_price_by_hour: Array,
    variable_om_rate: Array,
    fixed_om_rate: Array,
    capacity_installed: Array,
) -> Array:
    """Compute total operating costs per period.

    Formula:
    Cost_op_t = FuelCosts_t + VariableO&M_t + FixedO&M_t

    Args:
        heat_rate: Scalar or array broadcastable to (T, H) with heat rate.
        generation_by_hour: Array of shape (T, H) with hourly generation.
        fuel_price_by_hour: Scalar or array broadcastable to (T, H) with fuel
            price.
        variable_om_rate: Scalar or (T,) with variable O&M rate.
        fixed_om_rate: Scalar or (T,) with fixed O&M rate.
        capacity_installed: Array of shape (T,) with installed capacity.

    Returns:
        Array of shape (T,) with operating costs by period.
    """
    fuel = fuel_cost_ipps(heat_rate, generation_by_hour, fuel_price_by_hour)
    vom = variable_om_cost(variable_om_rate, generation_by_hour)
    fom = fixed_om_cost(fixed_om_rate, capacity_installed)
    return fuel + vom + fom


def ebit_from_components(
    total_revenue: Array,
    operating_costs_t: Array,
    depreciation: Optional[Array] = None,
    include_depreciation_in_ebit: bool = False,
) -> Array:
    """Compute EBIT per period from revenue and costs.

    Two conventions are supported:
    - If ``include_depreciation_in_ebit`` is True and ``depreciation`` is
      provided, then EBIT_t = Rev_t - OpCosts_t - Dep_t.
    - Otherwise, EBIT_t = Rev_t - OpCosts_t.

    Args:
        total_revenue: Array of shape (T,) with total revenue.
        operating_costs_t: Array of shape (T,) with operating costs.
        depreciation: Optional array of shape (T,) with depreciation.
        include_depreciation_in_ebit: Whether to subtract depreciation when
            forming EBIT.

    Returns:
        Array of shape (T,) with EBIT by period.
    """
    if include_depreciation_in_ebit and depreciation is not None:
        return total_revenue - operating_costs_t - depreciation
    return total_revenue - operating_costs_t


def financial_costs(
    debt_interest_rate: Array,       # scalar or (T,)
    debt_balance: Array,             # (T,)
    corporate_tax_rate: Array,       # scalar or (T,)
    earnings_before_interest_and_taxes: Array,  # (T,)
    depreciation: Array,             # (T,)
) -> Array:
    """Compute financial costs per period.

    Formula (as modeled here):
    Cost_fin_t = r_debt_t * D_t + tau_t * EBIT_t + Depreciation_t

    All scalar inputs are broadcast to shape (T,) as needed.

    Args:
        debt_interest_rate: Scalar or array of shape (T,) with debt rates.
        debt_balance: Array of shape (T,) with debt balances.
        corporate_tax_rate: Scalar or array of shape (T,) with corporate tax
            rates.
        earnings_before_interest_and_taxes: Array of shape (T,) with EBIT.
        depreciation: Array of shape (T,) with depreciation.

    Returns:
        Array of shape (T,) with financial costs by period.
    """
    r_d = np.broadcast_to(debt_interest_rate, debt_balance.shape)
    tau = np.broadcast_to(corporate_tax_rate, debt_balance.shape)
    return r_d * debt_balance + tau * earnings_before_interest_and_taxes + depreciation


def discount_factors_wacc(weighted_average_cost_of_capital: float, periods: int) -> Array:
    """Compute discount factors using WACC for end-of-period cash flows.

    Uses t = 1..T convention:
    df_t = 1 / (1 + r_wacc)^t

    Args:
        weighted_average_cost_of_capital: Scalar WACC.
        periods: Number of periods T.

    Returns:
        Array of shape (T,) with discount factors for t = 1..T.
    """
    return 1.0 / (1.0 + weighted_average_cost_of_capital) ** np.arange(1, periods + 1)


@dataclass(frozen=True)
class IppRevenueBreakdown:
    """Structured container for IPP revenue components by period.

    Attributes:
        energy: Array of shape (T,) with energy market revenue.
        capacity: Array of shape (T,) with capacity market revenue.
        ancillary_services: Array of shape (T,) with ancillary services
            revenue.
    """
    energy: Array
    capacity: Array
    ancillary_services: Array

    @property
    def total(self) -> Array:
        """Total revenue by period across all components.

        Returns:
            Array of shape (T,) with total revenue per period.
        """
        return self.energy + self.capacity + self.ancillary_services


def independent_power_producer_components(
    spot_price_by_hour: Array,                 # (T, H)
    generation_merchant_by_hour: Array,        # (T, H)
    power_purchase_agreement_price: Array,     # (K,) or (T, K)
    generation_contracted: Array,              # (T, K)
    capacity_available: Array,                 # (T,)
    capacity_price: Array,                     # (T,)
    availability_factor: Array,                # (T,)
    ancillary_service_price: Array,            # (T, S)
    ancillary_service_quantity: Array,         # (T, S)
    heat_rate: Array,                          # scalar or broadcastable to (T,H)
    generation_by_hour: Array,                 # (T, H)
    fuel_price_by_hour: Array,                 # (T, H)
    variable_om_rate: Array,                   # scalar or (T,)
    fixed_om_rate: Array,                      # scalar or (T,)
    capacity_installed: Array,                 # (T,)
) -> Dict[str, Array | IppRevenueBreakdown]:
    """Compute per-period components for the IPP objective.

    This function aggregates the component series required to evaluate the
    independent power producer objective, including revenue breakdown
    (energy, capacity, ancillary services) and total operating costs.

    Args:
        spot_price_by_hour: (T, H) hourly spot prices.
        generation_merchant_by_hour: (T, H) hourly merchant generation.
        power_purchase_agreement_price: (K,) or (T, K) PPA prices.
        generation_contracted: (T, K) contracted generation quantities.
        capacity_available: (T,) available capacity for capacity payments.
        capacity_price: (T,) capacity price.
        availability_factor: (T,) availability factor.
        ancillary_service_price: (T, S) price by ancillary service.
        ancillary_service_quantity: (T, S) quantity by ancillary service.
        heat_rate: Scalar or broadcastable to (T, H) heat rate.
        generation_by_hour: (T, H) total hourly generation.
        fuel_price_by_hour: (T, H) or broadcastable fuel price.
        variable_om_rate: Scalar or (T,) variable O&M rate.
        fixed_om_rate: Scalar or (T,) fixed O&M rate.
        capacity_installed: (T,) installed capacity.

    Returns:
        Dict with keys:
        - "revenue_breakdown": IppRevenueBreakdown of arrays (T,)
        - "operating_costs": Array (T,) with operating costs
    """
    rev_energy_t = energy_revenue(
        spot_price_by_hour, generation_merchant_by_hour,
        power_purchase_agreement_price, generation_contracted
    )
    rev_capacity_t = capacity_revenue(capacity_available, capacity_price, availability_factor)
    rev_as_t = ancillary_services_revenue(ancillary_service_price, ancillary_service_quantity)

    cost_op_t = operating_costs(
        heat_rate, generation_by_hour, fuel_price_by_hour,
        variable_om_rate, fixed_om_rate, capacity_installed
    )

    rev = IppRevenueBreakdown(energy=rev_energy_t, capacity=rev_capacity_t, ancillary_services=rev_as_t)

    return {
        "revenue_breakdown": rev,
        "operating_costs": cost_op_t,
    }


def independent_power_producer_objective(
    weighted_average_cost_of_capital: float,   # r_wacc
    initial_investment: float,                 # I0
    spot_price_by_hour: Array,
    generation_merchant_by_hour: Array,
    power_purchase_agreement_price: Array,
    generation_contracted: Array,
    capacity_available: Array,
    capacity_price: Array,
    availability_factor: Array,
    ancillary_service_price: Array,
    ancillary_service_quantity: Array,
    heat_rate: Array,
    generation_by_hour: Array,
    fuel_price_by_hour: Array,
    variable_om_rate: Array,
    fixed_om_rate: Array,
    capacity_installed: Array,
    debt_interest_rate: Array,
    debt_balance: Array,
    corporate_tax_rate: Array,
    depreciation: Array,
    earnings_before_interest_and_taxes: Optional[Array] = None,
    include_depreciation_in_ebit: bool = False,
) -> float:
    """Compute the IPP objective as the NPV of cash flows minus initial capex.

    Formula:
    Pi_IPP = sum_t [ (Rev_energy_t + Rev_capacity_t + Rev_AS_t
                       - Cost_op_t - Cost_fin_t) / (1 + r_wacc)^t ] - I0

    EBIT can be supplied directly; otherwise it is computed from components
    using the chosen convention on depreciation inclusion.

    Args:
        weighted_average_cost_of_capital: Scalar WACC used for discounting.
        initial_investment: Initial investment I0 (deducted after discounting
            the flow series).
        spot_price_by_hour: (T, H) hourly spot prices.
        generation_merchant_by_hour: (T, H) hourly merchant generation.
        power_purchase_agreement_price: (K,) or (T, K) PPA prices.
        generation_contracted: (T, K) contracted generation quantities.
        capacity_available: (T,) capacity eligible for capacity payments.
        capacity_price: (T,) capacity price.
        availability_factor: (T,) availability factor.
        ancillary_service_price: (T, S) price by ancillary service.
        ancillary_service_quantity: (T, S) quantity by ancillary service.
        heat_rate: Scalar or broadcastable to (T, H) heat rate.
        generation_by_hour: (T, H) total hourly generation.
        fuel_price_by_hour: (T, H) or broadcastable fuel price.
        variable_om_rate: Scalar or (T,) variable O&M rate.
        fixed_om_rate: Scalar or (T,) fixed O&M rate.
        capacity_installed: (T,) installed capacity.
        debt_interest_rate: Scalar or (T,) debt interest rate.
        debt_balance: (T,) outstanding debt balances.
        corporate_tax_rate: Scalar or (T,) corporate tax rate.
        depreciation: (T,) depreciation (used in financial costs and optional
            EBIT construction).
        earnings_before_interest_and_taxes: Optional (T,) provided EBIT.
        include_depreciation_in_ebit: Whether to subtract depreciation when
            constructing EBIT from components if EBIT is not provided.

    Returns:
        Float NPV of the IPP objective.
    """
    comps = independent_power_producer_components(
        spot_price_by_hour, generation_merchant_by_hour,
        power_purchase_agreement_price, generation_contracted,
        capacity_available, capacity_price, availability_factor,
        ancillary_service_price, ancillary_service_quantity,
        heat_rate, generation_by_hour, fuel_price_by_hour,
        variable_om_rate, fixed_om_rate, capacity_installed
    )

    rev_total_t = comps["revenue_breakdown"].total
    cost_op_t = comps["operating_costs"]

    # If EBIT is not supplied, derive it from components per the chosen convention
    if earnings_before_interest_and_taxes is None:
        ebit_t = ebit_from_components(
            total_revenue=rev_total_t,
            operating_costs_t=cost_op_t,
            depreciation=depreciation,
            include_depreciation_in_ebit=include_depreciation_in_ebit,
        )
    else:
        ebit_t = earnings_before_interest_and_taxes

    cost_fin_t = financial_costs(
        debt_interest_rate=debt_interest_rate,
        debt_balance=debt_balance,
        corporate_tax_rate=corporate_tax_rate,
        earnings_before_interest_and_taxes=ebit_t,
        depreciation=depreciation,
    )

    cash_flow_t = rev_total_t - cost_op_t - cost_fin_t

    T = rev_total_t.shape[0]
    df = discount_factors_wacc(weighted_average_cost_of_capital, T)
    npv = float((df * cash_flow_t).sum() - initial_investment)
    return npv


def risk_adjusted_utility(npv_per_scenario: Array, risk_aversion: float) -> float:
    """Compute mean-variance risk-adjusted utility from scenario NPVs.

    Formula:
    U_IPP = E[Pi_IPP] - (rho / 2) * Var[Pi_IPP]

    Args:
        npv_per_scenario: Array of shape (S,) with scenario NPVs.
        risk_aversion: Scalar risk aversion parameter rho.

    Returns:
        Float risk-adjusted utility value.
    """
    return float(np.mean(npv_per_scenario) - 0.5 * risk_aversion * np.var(npv_per_scenario, ddof=1))
