import numpy as np
from sklearn.base import check_array

### Overlap scores


def total_variation_distance(p, q):
    """Total variation distance between two probability distributions.
    .. math:: TV(p, q) = \frac{0.5}{N} \sum_{i=1}^N |p_i - q_i|
       Parameters
       ----------
       p : _type_
           _description_
       q : _type_
           _description_

       Returns
       -------
       _type_
           _description_
    """
    p = check_array(p, ensure_2d=False, dtype="numeric")
    q = check_array(q, ensure_2d=False, dtype="numeric")
    return float(0.5 * np.mean(np.abs(p - q)))


def normalized_total_variation(
    propensity_scores: np.ndarray, treatment_probability: float
):
    """Compute a renormalized total variation distance for a population.

    .. math:: nTV(p(A=1|X=x), P(A=1)) = \frac{0.5}{N}\sum_{i=1}^N |\frac{p(A=1|X=x_i)}{P(A=1)} - \frac{1 - p(A=1|X=x_i)}{1 - P(A=1)} |

    Parameters
    ----------
    propensity_scores : np.array
        _description_
    treatment_probability : float
        _description_

    Returns
    -------
    _type_
        _description_
    """
    normalized_tv = total_variation_distance(
        propensity_scores / treatment_probability,
        (1 - propensity_scores) / (1 - treatment_probability),
    )
    return normalized_tv
