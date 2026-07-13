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

    for key, var in node_state.items():
        if key.startswith(prefix_old):
            # Replace the old prefix with the new one
            new_key = prefix_new + key[len(prefix_old):] # Strip the old prefix and add the new one
            remapped[new_key] = var
        else:
            # Keep keys that don't follow the pattern
            remapped[key] = var

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
   
    # Map node names in state_dict to their base names
    state_by_base = {}
    for key in state_dict.keys():
        b = base_name(key)
        state_by_base.setdefault(b, []).append(key)


    nodes = target.nodes().subset(DynamicalSystem).not_subset(DynView).unique()
    missing_keys = []
    unexpected_keys = []
    
    # Remap outer keys
    for name, node in nodes.items():
        # Choose which state key to use for this node
        key_to_use = None

        # If exact name match exists
        if name in state_dict:
            key_to_use = name
            old_name= name
        else:
            # Try to find a base name match
            b = base_name(name)
            if b in state_by_base:
                keys = state_by_base[b]
                if len(keys) == 1:
                    key_to_use = keys[0]
                    old_name = keys[0]
                else:
                    # Ambigous match: multiple saved keys share the same base name
                    missing_keys.append(name)
                    print(f"[load_state_fixed] Ambiguous base name match for {name}: {keys}")
                    continue
            else:
                # No matching key at all
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

    