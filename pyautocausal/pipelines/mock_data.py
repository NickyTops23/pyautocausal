import numpy as np
import pandas as pd



def generate_mock_data(
    n_units: int = 100,
    n_periods: int = 1,
    n_treated: int = 50,
    n_pre_periods: int = 0,
    n_post_periods: int = 0,
    treatment_effect: float = 2.0,
    covariate_imbalance: float = 0.0,
    add_confounders: bool = False,
    add_colliders: bool = False,
    staggered_treatment: bool = False,
    random_seed: int = 42,
    omitted_variable_effect: float = 0.0  # New parameter for omitted variable bias
) -> pd.DataFrame:
    """
    Generate a mock dataset for testing causal inference methods.
    
    Args:
        n_units: Number of units (individuals, firms, etc.)
        n_periods: Total number of time periods (if 1, creates cross-sectional data)
        n_treated: Number of treated units
        n_pre_periods: Number of pre-treatment periods (for panel data)
        n_post_periods: Number of post-treatment periods (for panel data)
        treatment_effect: Size of the treatment effect
        covariate_imbalance: Degree of imbalance between treatment and control groups (0-1)
        add_confounders: Whether to add confounding variables
        add_colliders: Whether to add collider variables
        staggered_treatment: Whether treatment timing varies across units
        random_seed: Random seed for reproducibility
        omitted_variable_effect: Effect size of the omitted variable on the outcome
        
    Returns:
        DataFrame with columns:
            - id_unit: Unit identifier
            - t: Time period
            - treat: Treatment indicator
            - y: Outcome variable
            - x1, x2, ...: Covariates
            - relative_time: Time relative to treatment (for panel data)
            - post: Post-treatment indicator (for panel data)
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
    
    # For cross-sectional data (n_periods = 1)
    if n_periods == 1:
        # Generate unit IDs
        id_units = np.arange(n_units)
        
        # Assign treatment (first n_treated units are treated)
        treatment = np.zeros(n_units)
        treatment[:n_treated] = 1
        
        # Generate covariates with optional imbalance
        x1 = np.random.normal(0, 1, n_units)
        x2 = np.random.normal(0, 1, n_units)
        
        # Add imbalance to covariates if requested
        if covariate_imbalance > 0:
            x1[:n_treated] += covariate_imbalance * 2
            x2[:n_treated] -= covariate_imbalance * 1.5
        
        # Generate confounders if requested
        if add_confounders:
            # Confounder affects both treatment and outcome
            confounder = np.random.normal(0, 1, n_units)
            treatment_prob = 1 / (1 + np.exp(-confounder))
            treatment = np.random.binomial(1, treatment_prob, n_units)
            n_treated = int(treatment.sum())
        else:
            confounder = np.zeros(n_units)
        
        # Generate omitted variable
        omitted_variable = np.random.normal(0, 1, n_units)
        
        # Generate outcome with treatment effect and omitted variable
        noise = np.random.normal(0, 1, n_units)
        y = (0.5 + 0.3 * x1 - 0.2 * x2 + treatment_effect * treatment + 
             0.5 * confounder + omitted_variable_effect * omitted_variable + noise)
        
        # Generate collider if requested
        if add_colliders:
            # Collider is affected by both treatment and outcome
            collider = 0.5 * treatment + 0.5 * y + np.random.normal(0, 0.5, n_units)
        else:
            collider = np.zeros(n_units)
        
        # Create DataFrame
        df = pd.DataFrame({
            'id_unit': id_units,
            't': np.zeros(n_units),  # Single period
            'treat': treatment,
            'y': y,
            'x1': x1,
            'x2': x2,
            'confounder': confounder if add_confounders else np.zeros(n_units),
            'collider': collider if add_colliders else np.zeros(n_units)
        })
        
        # Add relative time (all 0 for cross-sectional)
        df['relative_time'] = 0
        df['post'] = 1
        
    # For panel data (n_periods > 1)
    else:
        # Generate data for each unit and period
        data = []
        
        # Determine treatment timing
        if staggered_treatment and n_pre_periods > 0:
            # Staggered treatment: units start treatment at different times
            treatment_starts = np.random.randint(
                n_pre_periods, n_pre_periods + max(1, n_post_periods - 1), 
                size=n_treated
            )
        else:
            # All treated units start treatment at the same time
            treatment_starts = np.full(n_treated, n_pre_periods)
        
        # Generate data for each unit
        for i in range(n_units):
            # Determine if unit is treated
            is_treated = i < n_treated
            
            # Unit-specific effects (fixed over time)
            unit_effect = np.random.normal(0, 1)
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
            omitted_variable_base = np.random.normal(0, 1)
            
            # Get treatment start time for this unit
            if is_treated:
                if staggered_treatment:
                    treatment_start = treatment_starts[min(i, len(treatment_starts)-1)]
                else:
                    treatment_start = n_pre_periods
            else:
                treatment_start = float('inf')  # Never treated
            
            # Generate data for each period
            for t in range(n_periods):
                # Time-varying components
                time_effect = 0.1 * t
                x1_noise = np.random.normal(0, 0.2)
                x2_noise = np.random.normal(0, 0.2)
                
                # Determine if unit is treated in this period
                treat = 1 if (is_treated and t >= treatment_start) else 0
                
                # Calculate relative time
                relative_time = t - treatment_start if is_treated else np.nan
                
                # Post-treatment indicator
                post = 1 if t >= treatment_start else 0
                
                # Generate outcome with treatment effect, omitted variable, and time trends
                # Add parallel trends for control and treated units
                base_trend = 0.1 * t
                
                # Add non-parallel pre-trends if confounders are present
                if add_confounders:
                    trend_modifier = 0.05 * t * confounder_base if is_treated else 0
                else:
                    trend_modifier = 0
                
                # Calculate outcome
                noise = np.random.normal(0, 1)
                x1 = x1_base + x1_noise
                x2 = x2_base + x2_noise
                confounder = confounder_base + np.random.normal(0, 0.2) if add_confounders else 0
                omitted_variable = omitted_variable_base + np.random.normal(0, 0.2)
                
                y = (0.5 + 0.3 * x1 - 0.2 * x2 + 
                     unit_effect + base_trend + trend_modifier +
                     treatment_effect * treat + 
                     omitted_variable_effect * omitted_variable +
                     0.5 * confounder + noise)
                
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
                    'relative_time': relative_time,
                    'post': post
                })
        
        # Create DataFrame
        df = pd.DataFrame(data)
    
    # Clean up columns based on what was requested
    if not add_confounders:
        df = df.drop(columns=['confounder'])
    
    if not add_colliders:
        df = df.drop(columns=['collider'])
    
    return df

