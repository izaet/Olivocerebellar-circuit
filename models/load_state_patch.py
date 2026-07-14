import brainpy as bp
import brainpy.helpers as helpers
from brainpy import dynsys
from brainpy.dynsys import DynamicalSystem, DynView
from typing import Dict, Callable
from brainpy.math.object_transform.base import StateLoadResult
import re


"""
Monkey patch for brainpy.helpers.load_state to fix the issue of not being able to replace/load network nodes when the names do not match.
BrainPy auto-generates unique names (PurkinjeCell2, PurkinjeCell4, …) via a global counter.

This function fixes the issue by matching the base names of the nodes, ignoring the digits attached to them.

States have the following structure, with 'PurkinjeCell13' being the brainpy.dyn.Neuron class (outer key).
"PurkinjeCell13.rho" and "PurkinjeCell13.V" are the Brainpy variables within that class (inner keys).

state['PurkinjeCell13'] == {
    "PurkinjeCell13.rho": ...,
    "PurkinjeCell13.V": ...,
    ...
}

"""

def base_name(name: str) -> str:
    """Strip digits from a BrainPy node name.

    e.g. 'PurkinjeCell2' -> 'PurkinjeCell'
    """
    return re.sub(r'\d+$', '', name)

def remap_inner_keys(old_name: str, new_name: str, node_state: Dict):
    """
    Renaming the inner keys of a node's state dictionary to match the new node name.

    args:
        old_name: The original name of the node (e.g., 'PurkinjeCell13').
        new_name: The new name of the node (e.g., 'PurkinjeCell2').
        node_state: The state dictionary of the node, where keys are in the format 'old_name.variable_name'.    

    returns:
        A new state dictionary with keys renamed to use the new node name.
    """

    if not isinstance(node_state, dict):
        return node_state

    remapped = {}
    prefix_old = old_name + "."
    prefix_new = new_name + "."
    for key, value in node_state.items():
            if isinstance(key, str) and key.startswith(prefix_old):
                remapped[prefix_new + key[len(prefix_old):]] = value
            else:
                remapped[key] = value

    return remapped

def load_state_fixed(target: DynamicalSystem, state_dict: Dict, **kwargs):
    """
    Load the state of a DynamicalSystem, remapping node names to match the current network digit suffixes. 
    The state dictionary values are loaded into the corresponding nodes of the target network.

    Args:
        target: The DynamicalSystem to load the state into.
        state_dict: A dictionary containing the state to load, with keys as node names and values as their states.
    returns:
        A StateLoadResult object containing lists of missing and unexpected keys. 
    
    """

    if not isinstance(state_dict, dict):
        return helpers.load_state(target, state_dict, **kwargs)

    # Clear stale runtime state before loading checkpoint state.
    try:
        helpers.reset_state(target)
        helpers.clear_input(target)
    except Exception:
        pass
   
    # Map node names in state_dict to their base names
    state_by_base = {}
    for key in state_dict.keys():
           if isinstance(key, str):
                state_by_base.setdefault(base_name(key), []).append(key)


    nodes = target.nodes().subset(DynamicalSystem).not_subset(DynView).unique()
    missing_keys = []
    unexpected_keys = []
    
    # Remap outer keys
    
    for name, node in nodes.items():
        key_to_use = None
        old_name = None

        # 1) Exact name match
        if name in state_dict:
            key_to_use = name
            old_name = name

        # 2) Base-name match
        else:
            candidates = state_by_base.get(base_name(name), [])
            if len(candidates) == 1:
                key_to_use = candidates[0]
                old_name = candidates[0]
            elif len(candidates) > 1:
                # Try to pick the most plausible candidate by looking for
                # variable names that are typical for this node.
                for cand in candidates:
                    cand_state = state_dict[cand]
                    if isinstance(cand_state, dict):
                        keys = set(cand_state.keys())
                        if any(k.split(".")[-1] in {"V", "rho", "spike", "I_OU", "I_PC", "I_CN", "I_stim"} for k in keys):
                            key_to_use = cand
                            old_name = cand
                            break
                if key_to_use is None:
                    missing_keys.append(name)
                    continue
            else:
                missing_keys.append(name)
                continue

        # Remap inner keys to new name
        node_state_raw = state_dict[key_to_use]
        node_state = remap_inner_keys(old_name, name, node_state_raw)

        # Load the state into the node
        r= node.load_state(node_state, **kwargs)
        if r is not None:
            missing, unexpected = r
            missing_keys.extend([f'{name}.{key}' for key in missing])
            unexpected_keys.extend([f'{name}.{key}' for key in unexpected])

    return StateLoadResult(missing_keys, unexpected_keys)

helpers.load_state = load_state_fixed
bp.load_state = load_state_fixed  

    