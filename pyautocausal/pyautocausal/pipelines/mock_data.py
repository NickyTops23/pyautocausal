import numpy as np
import pandas as pd


def generate_mock_data(
    n_units: int = 100,
    n_periods: int = 1,
    n_treated: int = 50,
    n_pre_periods: int = 0,
    n_post_periods: int = 0,
    treatment_effect: float = 2.0,
    noise_to_signal_ratio: float = 0.1,
    covariate_imbalance: float = 0.0,
    add_confounders: bool = False,
    add_colliders: bool = False,
    add_omitted_variable: bool = False,
    staggered_treatment: bool = False,
    random_seed: int = 42,
    omitted_variable_effect: float = 0.0
) -> pd.DataFrame:
    """
    Generate a simplified mock dataset for testing causal inference methods.
    
    Args:
        n_units: Number of units (individuals, firms, etc.)
        n_periods: Total number of time periods (if 1, creates cross-sectional data)
        n_treated: Number of treated units
        n_pre_periods: Number of pre-treatment periods (for panel data)
        n_post_periods: Number of post-treatment periods (for panel data)
        treatment_effect: Size of the treatment effect
        noise_to_signal_ratio: Ratio of noise to signal 
        covariate_imbalance: Degree of imbalance between treatment and control groups (0-1)
        add_confounders: Whether to add confounding variables
        add_colliders: Whether to add collider variables
        add_omitted_variable: Whether to add omitted variables
        staggered_treatment: Whether treatment timing varies across units
        random_seed: Random seed for reproducibility
        omitted_variable_effect: Effect size of the omitted variable on the outcome
        
    Returns:
        DataFrame with columns:
            - id_unit: Unit identifier
            - t: Time period
            - treat: Treatment indicator
            - y: Outcome variable
            - x1, x2: Covariates
            - relative_time: Time relative to treatment (for panel data)
            - post: Post-treatment indicator
            - noise: Noise component
    """
    np.random.seed(random_seed)
    
    # Validate inputs
    if n_periods > 1:
        # If pre and post periods are not specified, split periods evenly
        if n_pre_periods == 0 and n_post_periods == 0:
            n_pre_periods = n_periods // 2
            n_post_periods = n_periods - n_pre_periods
        # If only one is specified, calculate the other
        elif n_pre_periods == 0:
            n_pre_periods = n_periods - n_post_periods
        elif n_post_periods == 0:
            n_post_periods = n_periods - n_pre_periods
        # Check if they sum correctly
        if n_pre_periods + n_post_periods != n_periods:
            raise ValueError("For panel data, n_pre_periods + n_post_periods must equal n_periods")
    
    if n_treated > n_units:
        raise ValueError("Number of treated units cannot exceed total units")
    
    # Generate data for each unit and period
    data = []
    
    # Determine treatment timing (only matters for panel data with n_periods > 1)
    if staggered_treatment and n_pre_periods > 0 and n_periods > 1:
        # Staggered treatment: units start treatment at different times
        treatment_starts = np.random.randint(
            n_pre_periods, n_pre_periods + max(1, n_post_periods - 1), 
            size=n_treated
        )
    else:
        # All treated units start treatment at the same time
        treatment_starts = np.full(n_treated, n_pre_periods if n_periods > 1 else 0)
    
    # Generate data for each unit
    for i in range(n_units):
        # Determine if unit is treated
        is_treated = i < n_treated
        
        # Generate covariates
        x1_base = np.random.normal(0, 1)
        x2_base = np.random.normal(0, 1)
        
        # Add imbalance to covariates if requested
        if covariate_imbalance > 0 and is_treated:
            x1_base += covariate_imbalance * 2
            x2_base -= covariate_imbalance * 1.5
        
        # Generate confounder if requested
        if add_confounders:
            confounder_base = np.random.normal(0, 1)
            # Confounder affects treatment probability
            if not staggered_treatment:
                is_treated = np.random.binomial(1, 1 / (1 + np.exp(-confounder_base)))
        else:
            confounder_base = 0
        
        # Generate omitted variable
        if add_omitted_variable:
            omitted_variable_base = np.random.normal(0, 1)
        else:
            omitted_variable_base = 0
        
        # Get treatment start time for this unit
        if is_treated:
            if staggered_treatment and n_periods > 1:
                treatment_start = treatment_starts[min(i, len(treatment_starts)-1)]
            else:
                treatment_start = n_pre_periods if n_periods > 1 else 0
        else:
            treatment_start = float('inf')  # Never treated
        
        # For calculating noise std
        unit_signals = []
        
        # Generate data for each period (or just once for cross-sectional)
        periods_to_generate = n_periods if n_periods > 1 else 1
        
        for t in range(periods_to_generate):
            # Determine if unit is treated in this period
            treat = 1 if (is_treated and t >= treatment_start) else 0
            
            # Calculate relative time
            relative_time = t - treatment_start if is_treated else np.nan
            
            # Post-treatment indicator
            post = 1 if t >= treatment_start else 0
            
            # Calculate features
            x1 = x1_base
            x2 = x2_base
            confounder = confounder_base
            omitted_variable = omitted_variable_base
            
            # Calculate signal component (simplified - no unit or time fixed effects)
            signal = (1 + 1 * x1 + 1 * x2 + 
                  treatment_effect * treat + 
                  omitted_variable_effect * omitted_variable +
                  1 * confounder)
            
            unit_signals.append(signal)
        
        # Calculate noise std based on signal variance
        signal_std = np.std(unit_signals) if len(unit_signals) > 1 else np.abs(unit_signals[0])
        noise_std = signal_std * noise_to_signal_ratio
        
        # Add data for each period with the calculated noise
        for t, signal in enumerate(unit_signals):
            # Recalculate treatment indicators
            treat = 1 if (is_treated and t >= treatment_start) else 0
            relative_time = t - treatment_start if is_treated else np.nan
            post = 1 if t >= treatment_start else 0
            
            # Add noise to signal
            noise = np.random.normal(0, noise_std)
            y = signal + noise
            
            # Generate collider if requested
            if add_colliders:
                collider = 0.5 * treat + 0.5 * y + np.random.normal(0, 0.5)
            else:
                collider = 0
            
            # Add to data
            data.append({
                'id_unit': i,
                't': t,
                'treat': treat,
                'y': y,
                'x1': x1,
                'x2': x2,
                'confounder': confounder,
                'collider': collider,
                'post': post,
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Clean up columns based on what was requested
    if not add_confounders:
        df = df.drop(columns=['confounder'])
    
    if not add_colliders:
        df = df.drop(columns=['collider'])
    
    return df

