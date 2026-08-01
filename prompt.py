SYSTEM_PROMPT = """
    You are a claims-denial risk assistant for healthcare billing analysts.
    Your job is to explain, in plain language, why a specific insurance claim is
    flagged as high risk for denial, based ONLY on the field values and risk
    drivers provided. Follow these rules strictly:

    - Use ONLY the facts in the claim data below. Do NOT invent diagnoses,
    amounts, dates, payer names, or any detail not present.
    - Write for a busy analyst: plain language, no insurance jargon that would
    need looking up (e.g., say "prior approval from the insurer" not "prior auth").
    - Always include exactly ONE specific, actionable next step.
    - Always state that this is a risk estimate, not a guarantee the claim will
    be denied.
    - Keep it to 2-3 sentences. No bullet points, no headers.
"""

USER_PROMPT = """
    Claim ID: {claim_id}
    Model risk score (0-1, higher = more likely denied): {risk_score}

    Claim facts (only use fields that are present/true):
    - Prior approval required but not obtained: {prior_auth_missing}
    - Required documentation missing: {missing_documentation}
    - Required referral missing: {missing_referral}
    - Provider is outside the insurer's network: {out_of_network}
    - Patient eligibility not verified before service: {unverified_eligibility}
    - Uncovered amount (billed minus expected payment): {patient_responsibility_delta}
    - Filed later than the allowed window (e.g., >30 days): {timely_filing_flag}

    Write the 2-3 sentence explanation now.
"""

PROMPT = SYSTEM_PROMPT + "\n" + USER_PROMPT


def yn(v):
    """Convert a 0/1 flag into readable YES/no."""
    return "YES" if int(v) == 1 else "no"


def get_prompt(row):
    user_filled = PROMPT.format(
        risk_score=f"{row['predicted_denial_probability']:.2f}",
        prior_auth_missing=yn(row.get("features_unauthorized_proc_flag", 0)),
        missing_documentation=yn(row.get("features_missing_documentation_flag", 0)),
        missing_referral=yn(row.get("features_referral_required_but_not_present", 0)),
        out_of_network=yn(row.get("features_out_of_network_flag", 0)),
        unverified_eligibility=yn(row.get("features_eligibility_not_verified_flag", 0)),
        patient_responsibility_delta=f"${int(row['total_billed'] - row['expected_payment'])}",
        timely_filing_flag=yn(row.get("features_days_g30", 0)),
    )
    return user_filled
