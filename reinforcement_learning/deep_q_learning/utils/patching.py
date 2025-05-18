#!/usr/bin/env python3
"""
Module for patching the DQNAgent for continuous training without warmup reset.
"""

def patch_dqn_for_continuous_training(dqn_agent):
    """
    Patch DQNAgent to allow continuous training without warmup reset.
    This modifies the DQNAgent instance to skip warmup on subsequent fit() calls.
    """
    # Store original fit method
    original_fit = dqn_agent.fit
    
    # Flag to track if we've already done the warmup
    dqn_agent._warmup_done = False
    
    # Define patched fit method
    def patched_fit(env, nb_steps, **kwargs):
        # If we've already done warmup in a previous fit call
        if dqn_agent._warmup_done:
            # Temporarily set warmup steps to 0
            original_warmup_steps = dqn_agent.nb_steps_warmup
            dqn_agent.nb_steps_warmup = 0
            
            # Call original fit
            result = original_fit(env, nb_steps, **kwargs)
            
            # Restore original warmup steps
            dqn_agent.nb_steps_warmup = original_warmup_steps
            return result
        else:
            # First time training, do normal warmup
            result = original_fit(env, nb_steps, **kwargs)
            
            # Mark warmup as done for future fit calls
            dqn_agent._warmup_done = True
            return result
            
    # Replace the fit method with our patched version
    dqn_agent.fit = patched_fit.__get__(dqn_agent)
    
    return dqn_agent
