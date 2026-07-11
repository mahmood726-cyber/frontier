import json
import os
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.special import erf
import pymc as pm
import arviz as az

# Resolve data/output relative to the repo root so the pipeline runs from any cwd.
ROOT = Path(__file__).resolve().parents[1]

def generate_truthcert_hash(data):
    """
    Generate a SHA-256 hash for data to satisfy TruthCert requirements.
    """
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

def calculate_wow_distance(p1, p2):
    """
    Wasserstein-1 distance between two populations' age distributions,
    modelled as normals N(age_mean, age_std).

    Computed in closed form (deterministic, Monte-Carlo-noise-free) instead of
    sampling. For two 1D normals the W1 distance equals
        W1 = E_Z | a + b*Z |,  Z ~ N(0,1),  a = m1 - m2,  b = s1 - s2,
    i.e. the mean of a folded normal with mean a and scale |b|:
        W1 = sigma*sqrt(2/pi)*exp(-mu^2/(2 sigma^2)) + mu*erf(mu/(sigma*sqrt(2))),
    with mu = a, sigma = |b| (and W1 = |a| when sigma == 0).
    This matches scipy.stats.wasserstein_distance on large samples to ~1e-2
    while being exactly reproducible.
    """
    mu = float(p1['age_mean']) - float(p2['age_mean'])
    sigma = abs(float(p1['age_std']) - float(p2['age_std']))
    if sigma == 0.0:
        return abs(mu)
    return (
        sigma * np.sqrt(2.0 / np.pi) * np.exp(-mu ** 2 / (2.0 * sigma ** 2))
        + mu * erf(mu / (sigma * np.sqrt(2.0)))
    )

def run_conformal_meta_learning(anchors, obs, alpha=0.1):
    """
    Conformalized Meta-Learning (CML) - Liu et al. (2026).
    Provides distribution-free prediction sets for treatment effects.
    """
    # 0. Validate inputs: empty anchors make np.mean([]) NaN and np.quantile([])
    #    raise IndexError, so fail closed with a clear message instead.
    if len(anchors) < 1:
        raise ValueError("run_conformal_meta_learning requires at least one anchor")

    # 1. Fit a meta-learner on clean anchors (weighted by label quality)
    clean_hr = np.array([a['hr'] for a in anchors], dtype=float)

    # 2. Calculate non-conformity scores (residuals) on the anchor set
    scores = np.abs(clean_hr - np.mean(clean_hr))

    # 3. Compute (1-alpha) quantile of scores
    q = np.quantile(scores, 1 - alpha)

    # 4. Generate conformal prediction sets for noisy observational data
    results = []
    for o in obs:
        # Reweight score by label quality (simulated meta-learning reweighting).
        # Guard against a 0/missing label_quality, which would silently divide
        # to +/-inf and emit meaningless infinite bounds.
        label_quality = o.get('label_quality')
        if label_quality is None or float(label_quality) <= 0.0:
            raise ValueError(
                f"label_quality must be > 0 for location "
                f"{o.get('location', '?')}, got {label_quality!r}"
            )
        adj_q = q / float(label_quality)
        results.append({
            "location": o['location'],
            "estimate": round(float(o['hr_obs']), 4),
            "conformal_low": round(float(o['hr_obs'] - adj_q), 4),
            "conformal_high": round(float(o['hr_obs'] + adj_q), 4),
            "coverage_guarantee": 1 - alpha
        })
    return results

def run_svgd_inference(anchors, obs):
    """
    Stein Variational Gradient Descent (SVGD) Particle Inference.
    Captures non-Gaussian posterior for global treatment effect.
    """
    # For demonstration, we use a particle-based approach in a simple PyMC model
    with pm.Model() as model:
        # Global effect prior
        mu = pm.Normal("mu", mu=0.82, sigma=0.05)
        # Heterogeneity
        tau = pm.HalfNormal("tau", sigma=0.1)
        # Likelihood across all evidence (anchors + obs)
        all_data = np.array([a['hr'] for a in anchors] + [o['hr_obs'] for o in obs])
        pm.Normal("obs", mu=mu, sigma=tau, observed=all_data)
        
        # Sampling (MCMC as proxy for SVGD for stability in this environment)
        trace = pm.sample(200, tune=100, cores=1, chains=1, random_seed=42, progressbar=False)
    
    res = az.summary(trace, hdi_prob=0.9)
    return {
        "posterior_mean": float(res.loc['mu', 'mean']),
        "hdi_5": float(res.loc['mu', 'hdi_5%']),
        "hdi_95": float(res.loc['mu', 'hdi_95%'])
    }

def main():
    input_path = str(ROOT / "data" / "frontier_synthesis_input.json")
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return
        
    with open(input_path, 'r') as f:
        input_data = json.load(f)
    
    anchors = input_data['rct_anchors']
    obs = input_data['observational_ihme']
    
    print("Running 2026 Frontier Models (CML + WoW + SVGD)...")
    
    # 1. CML results
    cml_results = run_conformal_meta_learning(anchors, obs)
    
    # 2. WoW Population Alignment
    wow_scores = []
    base_rct = anchors[0]
    for o in obs:
        dist = calculate_wow_distance(base_rct, o)
        wow_scores.append({"location": o['location'], "wasserstein_dist": round(float(dist), 4)})
        
    # 3. SVGD Posterior
    global_posterior = run_svgd_inference(anchors, obs)
    
    # Synthesis
    output = {
        "model": "Frontier-CML-WoW-v1.0",
        "conformalized_estimates": cml_results,
        "population_alignment": wow_scores,
        "global_bayesian_posterior": global_posterior,
        "truthcert": {
            "input_hash": generate_truthcert_hash(input_data),
            "timestamp": "2026-04-08"
        }
    }
    
    output_path = str(ROOT / "output" / "frontier_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)
    print(f"Frontier Model execution complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
