"""Engineered outcome reward and signed plasticity, not a biological learning claim."""
import numpy as np


def cutting_reward(action, metrics):
    # No cue, alignment label, or target action is accepted by this function.
    required = ["prefracture_slip_mm", "peak_blade_force_model_units",
                "knife_pad_force_model_units"]
    if any(not np.isfinite(metrics[k]) or metrics[k] < 0 for k in required):
        raise ValueError("Invalid physical metrics")
    if metrics["knife_pad_force_model_units"] > 0.001:
        return -1.0, "blade-foot contact"
    if action == "hold":
        return 0.0, "no cutting attempt"
    if action != "chop":
        raise ValueError("Unknown action")
    if not metrics["seam_broken"]:
        return -1.0, "cut missed or seam intact"
    first, cut = metrics["first_pad_contact_s"], metrics["cut_time_s"]
    if first is None or cut is None or first >= cut or (metrics["pad_force_at_fracture_model_units"] or 0) <= 0.001:
        return -1.0, "cut without holding contact"
    if metrics["prefracture_slip_mm"] > 0.12:
        return -1.0, "excessive slip"
    if metrics["peak_blade_force_model_units"] > 50:
        return -1.0, "excessive blade force"
    return 1.0, "held cut"


def update_weights(previous, original, eligibility, reward, action, rate=0.15):
    # Lower MBON11 excitation favors chop in the existing fixed decoder.
    # A failed chop reverses that update. Holding has no action credit.
    if action != "chop" or reward == 0:
        return previous.copy()
    return np.clip(previous*np.exp(-rate*reward*eligibility),
                   0.1*original, 2*original).astype(previous.dtype)
