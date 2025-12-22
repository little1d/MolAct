from typing import Literal
from agentfly.tools import tool
import os

# Set TDC cache path early, before any TDC imports
# This ensures Ray workers can access the cache even if env vars aren't passed via runtime_env
if "TDC_CACHE_PATH" not in os.environ:
    # Try common paths
    default_paths = [
        "/mnt/shared-storage-user/yangzhuo/main/projects/agentrl/AgentFly/verl/oracle",
        os.path.expanduser("~/.cache/tdc"),
        os.path.join(os.path.expanduser("~"), ".cache", "tdc"),
    ]
    for path in default_paths:
        if os.path.exists(path):
            os.environ["TDC_CACHE_PATH"] = path
            break

# Also set PYTDC_CACHE if TDC uses that name
if "TDC_CACHE_PATH" in os.environ and "PYTDC_CACHE" not in os.environ:
    os.environ["PYTDC_CACHE"] = os.environ["TDC_CACHE_PATH"]

# Fix for RDKit 2024.x compatibility: TDC expects rdkit.six which was removed
# Create a compatibility shim for rdkit.six before TDC imports
# This must be done before any TDC imports
try:
    import sys
    # Check if rdkit.six module already exists
    if 'rdkit.six' not in sys.modules:
        try:
            import rdkit
            # Check if rdkit.six exists (it doesn't in RDKit 2024.x)
            if not hasattr(rdkit, 'six'):
                # Try to import six package first (most compatible)
                try:
                    import six
                    # Make six available as rdkit.six
                    rdkit.six = six
                    sys.modules['rdkit.six'] = six
                except ImportError:
                    print(f"[DEBUG] six package not found, creating minimal compatibility shim")
                    # If six package is not installed, create a minimal compatibility shim
                    # This provides the minimal interface that TDC's oracle.py expects
                    class SixCompat:
                        @staticmethod
                        def iteritems(d):
                            """Python 3 compatibility: dict.items() returns an iterator"""
                            return d.items()
                        @staticmethod
                        def itervalues(d):
                            """Python 3 compatibility: dict.values() returns an iterator"""
                            return d.values()
                        @staticmethod
                        def iterkeys(d):
                            """Python 3 compatibility: dict.keys() returns an iterator"""
                            return d.keys()
                        @staticmethod
                        def string_types():
                            """Python 3 compatibility: str is the only string type"""
                            return str
                        @staticmethod
                        def integer_types():
                            """Python 3 compatibility: int is the only integer type"""
                            return int
                    # Create the module and attach to rdkit
                    rdkit.six = SixCompat()
                    sys.modules['rdkit.six'] = SixCompat()
        except ImportError:
            # RDKit not installed, will fail later when TDC tries to import
            pass
except Exception as e:
    # Silently fail, will be caught later when TDC tries to import
    print(f"[WARNING] Failed to create rdkit.six compatibility shim: {e}")
    pass

# Lazy loading of TDC oracles to avoid import errors if TDC is not installed
_ORACLES = {}
# Common alias mapping (chemcotbench uses drd2/jnk3/gsk3b)
_ALIAS_MAP = {
    "drd2": "drd",
    "jnk3": "jnk",
    "gsk3b": "gsk",
}

def _get_oracle(prop: str):
    """Lazy load oracle to avoid import errors."""
    if prop in _ALIAS_MAP:
        prop = _ALIAS_MAP[prop]
    if prop not in _ORACLES:
        try:
            # TDC_CACHE_PATH should already be set at module level, but double-check
            if "TDC_CACHE_PATH" not in os.environ:
                default_cache = "/mnt/shared-storage-user/yangzhuo/main/projects/agentrl/AgentFly/verl/oracle"
                if os.path.exists(default_cache):
                    os.environ["TDC_CACHE_PATH"] = default_cache
                    if "PYTDC_CACHE" not in os.environ:
                        os.environ["PYTDC_CACHE"] = default_cache
            
            from tdc import Oracle
            oracle_map = {
                "logp": "logp",
                "qed": "qed",
                "drd": "drd2",
                "jnk": "jnk3",
                "gsk": "gsk3b",
                "sa": "sa",
            }
            if prop in oracle_map:
                tdc_name = oracle_map[prop]
                _ORACLES[prop] = Oracle(tdc_name)
        except ImportError as e:
            print(f"[ERROR] TDC import failed for '{prop}': {e}")
            import traceback
            traceback.print_exc()
            return None
        except (AttributeError, RuntimeError) as e:
            # NumPy compatibility or other initialization errors
            print(f"[ERROR] TDC Oracle initialization failed for '{prop}': {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None
        except ValueError as e:
            # scikit-learn version compatibility issue with pickle files
            error_str = str(e)
            if "incompatible dtype" in error_str or "pickle" in error_str.lower():
                print(f"[ERROR] scikit-learn version incompatibility for '{prop}': {e}")
                print(f"[INFO] The pickle file for '{prop}' was saved with an older scikit-learn version.")
                print(f"[INFO] Solution: Install compatible scikit-learn version: pip install 'scikit-learn<1.3'")
                return None
            else:
                # Re-raise if it's a different ValueError
                raise
        except Exception as e:
            # Catch any other unexpected errors and print them for debugging
            print(f"[ERROR] Unexpected error loading TDC Oracle for '{prop}': {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None
    return _ORACLES.get(prop)


@tool(name="chem_oracle_score", description="Get a property score from TDC oracles for drug discovery.")
def oracle_score(smiles: str, prop: str):
    """
    Get a property score from pre-defined TDC oracles.
    Supported props: logp, qed, drd (alias: drd2), jnk (alias: jnk3), gsk (alias: gsk3b), sa
    """
    if not smiles or not isinstance(smiles, str):
        return {"observation": "Error: empty or invalid SMILES input", "valid": 0}
    
    smiles = smiles.strip()
    
    supported_props = ["logp", "qed", "drd", "jnk", "gsk", "sa", "drd2", "jnk3", "gsk3b"]
    if prop not in supported_props:
        return {
            "observation": f"Error: unsupported property '{prop}'. Supported: {', '.join(supported_props)}",
            "valid": 0
        }
    
    if prop in _ALIAS_MAP:
        prop = _ALIAS_MAP[prop]
    oracle = _get_oracle(prop)
    if oracle is None:
        return {
            "observation": f"Error: TDC oracle for '{prop}' not available. Make sure TDC is installed.",
            "valid": 0
        }
    
    try:
        score = float(oracle(smiles))
        
        # Add interpretation for some properties
        interpretation = ""
        if prop == "qed":
            if score >= 0.7:
                interpretation = " (high drug-likeness)"
            elif score >= 0.4:
                interpretation = " (moderate drug-likeness)"
            else:
                interpretation = " (low drug-likeness)"
        elif prop in ["drd", "jnk", "gsk"]:
            if score >= 0.5:
                interpretation = " (likely active)"
            else:
                interpretation = " (likely inactive)"
        
        return {
            "observation": f"ok: {prop} = {score:.4f}{interpretation}",
            "valid": 1,
            "prop": prop,
            "score": round(score, 4)
        }
        
    except Exception as e:
        return {
            "observation": f"Error calculating {prop}: {str(e)}",
            "valid": 0
        }


@tool(name="chem_multi_objective_score", description="Calculate multiple property scores at once for multi-objective optimization.")
def multi_objective_score(smiles: str, props: str = "qed,sa"):
    """
    Calculate multiple property scores at once.
    Props should be comma-separated (e.g., 'qed,sa,logp').
    """
    if not smiles or not isinstance(smiles, str):
        return {"observation": "Error: empty or invalid SMILES input", "valid": 0}
    
    smiles = smiles.strip()
    prop_list = [p.strip().lower() for p in props.split(",")]
    
    supported_props = ["logp", "qed", "drd", "jnk", "gsk", "sa"]
    invalid_props = [p for p in prop_list if p not in supported_props]
    if invalid_props:
        return {
            "observation": f"Error: unsupported properties: {invalid_props}. Supported: {supported_props}",
            "valid": 0
        }
    
    scores = {}
    errors = []
    
    for prop in prop_list:
        if prop in _ALIAS_MAP:
            prop = _ALIAS_MAP[prop]
        oracle = _get_oracle(prop)
        if oracle is None:
            errors.append(f"{prop}: oracle not available")
            continue
        try:
            score = float(oracle(smiles))
            scores[prop] = round(score, 4)
        except Exception as e:
            errors.append(f"{prop}: {str(e)}")
    
    if not scores:
        return {
            "observation": f"Error: no properties could be calculated. Errors: {errors}",
            "valid": 0
        }
    
    # Create observation summary
    score_strs = [f"{k}={v:.3f}" for k, v in scores.items()]
    obs = f"ok: {', '.join(score_strs)}"
    if errors:
        obs += f" (errors: {errors})"
    
    return {
        "observation": obs,
        "valid": 1,
        "scores": scores,
        "errors": errors if errors else None
    }
