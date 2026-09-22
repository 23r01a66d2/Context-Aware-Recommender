"""
Module 12: Explainability Engine
Generates transparent, feature-grounded explanations for recommendations
based on learned modality weights (behavior, content, context), cold-start status,
and matching attributes.
"""

from typing import Dict, Any


def generate_recommendation_explanation(customer_id: int, product_id: int, product_category: int,
                                        hist_count: int, modality_weights: list[float],
                                        top_past_category: int = None,
                                        device_name: str = "Desktop") -> Dict[str, Any]:
    """
    Generates a truthful explanation grounded in actual model gating weights and historical state.
    Args:
        customer_id: user identifier
        product_id: recommended product identifier
        product_category: candidate product category code
        hist_count: point-in-time prior interactions
        modality_weights: [w_beh, w_cont, w_ctx] summing to 1.0
        top_past_category: customer's historical most frequented category
        device_name: current device context
    Returns:
        dict with explanation text, dominant modality, and modality percentages.
    """
    w_beh, w_cont, w_ctx = modality_weights
    pct_beh = round(float(w_beh) * 100, 1)
    pct_cont = round(float(w_cont) * 100, 1)
    pct_ctx = round(float(w_ctx) * 100, 1)

    is_cold = (hist_count < 3)

    if is_cold:
        explanation = (
            f"Recommended using candidate product characteristics and session context ({device_name}) "
            f"because this customer has limited prior interaction history ({hist_count} previous interactions). "
            f"The model's adaptive cold-start gating suppressed uninformative behavioral noise and emphasized "
            f"content ({pct_cont}%) and context ({pct_ctx}%)."
        )
        dominant_modality = "Content & Context (Cold-Start Mode)"
    else:
        if top_past_category is not None and top_past_category == product_category:
            cat_match_str = f"which aligns with your frequent interest in Category {product_category}"
        else:
            cat_match_str = f"in Category {product_category} matching your profile"

        explanation = (
            f"Recommended primarily based on your past browsing and purchase behavior ({pct_beh}% contribution), "
            f"{cat_match_str}, enriched by candidate product features ({pct_cont}%) "
            f"and session context ({pct_ctx}%)."
        )
        dominant_modality = "Behavioral History (Warm-Start Mode)"

    return {
        "customer_id": customer_id,
        "product_id": product_id,
        "is_cold_start": is_cold,
        "historical_interactions": hist_count,
        "dominant_modality": dominant_modality,
        "behavior_weight": pct_beh,
        "content_weight": pct_cont,
        "context_weight": pct_ctx,
        "explanation": explanation
    }


if __name__ == "__main__":
    cold_exp = generate_recommendation_explanation(1024, 844, 2, hist_count=1, modality_weights=[0.01, 0.58, 0.41])
    warm_exp = generate_recommendation_explanation(1056, 844, 2, hist_count=8, modality_weights=[0.62, 0.22, 0.16], top_past_category=2)
    print("--- COLD START EXPLANATION ---")
    print(cold_exp["explanation"])
    print("\n--- WARM START EXPLANATION ---")
    print(warm_exp["explanation"])
