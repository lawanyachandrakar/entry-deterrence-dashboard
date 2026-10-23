
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# ENTRY DETERRENCE DASHBOARD
# WAI Project | Games of Incomplete Information
# All numerical assumptions are illustrative.
# ==========================================================

st.set_page_config(
    page_title="Entry Deterrence Dashboard",
    page_icon="📊",
    layout="wide"
)

# ------------------ PAGE DESIGN ---------------------------

st.title("The Entry Deterrence Game")
st.subheader(
    "AI-Assisted Analysis of Startup Market Entry "
    "Under Incomplete Information"
)

st.markdown(
    """
    This dashboard examines how a startup's market-entry decision
    changes when the incumbent's competitive behaviour is uncertain.
    Adjust the assumptions in the sidebar to explore alternative
    scenarios.
    """
)

st.info(
    "Academic simulation: all payoffs and signal probabilities are "
    "hypothetical. They are not estimates of actual company behaviour. "
    "The current version is a preliminary expected-payoff model, "
    "not a fully solved Bayesian equilibrium."
)

# ------------------ SIDEBAR CONTROLS ----------------------

st.sidebar.header("Model Controls")

st.sidebar.markdown("### 1. Beliefs about the incumbent")

prior_aggressive = st.sidebar.slider(
    "Prior probability of an aggressive incumbent",
    min_value=0,
    max_value=100,
    value=40,
    step=5
) / 100

signal = st.sidebar.selectbox(
    "Observed signal",
    [
        "No new signal",
        "Warning signal",
        "Reassuring signal"
    ]
)

st.sidebar.markdown("### 2. Entry economics")

entry_cost = st.sidebar.slider(
    "Fixed entry cost (payoff units)",
    min_value=0.0,
    max_value=5.0,
    value=2.0,
    step=0.5
)

accommodation_payoff = st.sidebar.slider(
    "Entrant payoff if accommodated, before entry cost",
    min_value=1.0,
    max_value=8.0,
    value=4.0,
    step=0.5
)

fight_payoff = st.sidebar.slider(
    "Entrant payoff if the incumbent fights, before entry cost",
    min_value=-8.0,
    max_value=0.0,
    value=-3.0,
    step=0.5
)

st.sidebar.markdown("### 3. Signal assumptions")

signal_aggressive = st.sidebar.slider(
    "P(warning signal | aggressive incumbent)",
    min_value=0.05,
    max_value=0.95,
    value=0.80,
    step=0.05
)

signal_accommodating = st.sidebar.slider(
    "P(warning signal | accommodating incumbent)",
    min_value=0.05,
    max_value=0.95,
    value=0.30,
    step=0.05
)

# ------------------ BAYESIAN BELIEF UPDATING ---------------

def update_belief(prior, observed_signal, p_warning_aggressive,
                  p_warning_accommodating):
    """Calculate posterior probability using Bayes' rule."""

    if observed_signal == "No new signal":
        return prior

    if observed_signal == "Warning signal":
        likelihood_aggressive = p_warning_aggressive
        likelihood_accommodating = p_warning_accommodating
    else:
        likelihood_aggressive = 1 - p_warning_aggressive
        likelihood_accommodating = 1 - p_warning_accommodating

    numerator = prior * likelihood_aggressive

    denominator = (
        prior * likelihood_aggressive
        + (1 - prior) * likelihood_accommodating
    )

    if denominator == 0:
        return prior

    return numerator / denominator


posterior_aggressive = update_belief(
    prior_aggressive,
    signal,
    signal_aggressive,
    signal_accommodating
)

# ------------------ EXPECTED PAYOFF MODEL ------------------

def expected_entry_payoff(
    probability_aggressive,
    cost,
    payoff_fight,
    payoff_accommodate
):
    """
    Preliminary model:
    assumes the incumbent fights if aggressive and accommodates
    otherwise. This response rule is an assumption, not a
    strategic equilibrium result.
    """

    return (
        probability_aggressive * payoff_fight
        + (1 - probability_aggressive) * payoff_accommodate
        - cost
    )


def decision_rule(payoff):
    if payoff > 0:
        return "ENTER"
    elif payoff < 0:
        return "STAY OUT"
    return "INDIFFERENT"


expected_payoff = expected_entry_payoff(
    posterior_aggressive,
    entry_cost,
    fight_payoff,
    accommodation_payoff
)

decision = decision_rule(expected_payoff)

# ------------------ KEY METRICS ----------------------------

st.markdown("## Executive Dashboard")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Prior aggression probability",
    f"{prior_aggressive:.0%}"
)

m2.metric(
    "Updated aggression probability",
    f"{posterior_aggressive:.1%}",
    delta=f"{(posterior_aggressive - prior_aggressive):+.1%}"
)

m3.metric(
    "Expected entry payoff",
    f"{expected_payoff:.2f}"
)

m4.metric(
    "Preliminary decision",
    decision
)

st.caption(
    "The decision compares the expected entry payoff with a "
    "stay-out payoff of zero. It does not account for the value "
    "of waiting, strategic signalling by the incumbent, or "
    "type-dependent optimal responses."
)

st.divider()

# ------------------ CHART 1 --------------------------------

st.markdown("## 1. Expected Payoff vs. Aggression Probability")

probabilities = np.linspace(0, 1, 21)

chart1_df = pd.DataFrame({
    "Aggressive probability": probabilities,
    "Expected entry payoff": [
        expected_entry_payoff(
            p, entry_cost, fight_payoff, accommodation_payoff
        )
        for p in probabilities
    ]
})

fig1 = px.line(
    chart1_df,
    x="Aggressive probability",
    y="Expected entry payoff",
    markers=True,
    title="How Perceived Incumbent Aggression Changes Entry Payoff"
)

fig1.add_hline(
    y=0,
    line_dash="dash",
    annotation_text="Stay-out payoff"
)

fig1.add_vline(
    x=posterior_aggressive,
    line_dash="dot",
    annotation_text="Current belief"
)

fig1.update_xaxes(tickformat=".0%")
fig1.update_layout(template="plotly_white")

st.plotly_chart(fig1, use_container_width=True)

st.write(
    "Interpretation: increasing the perceived probability of "
    "aggression lowers expected entry payoff when fighting is "
    "less attractive than accommodation, holding other assumptions "
    "constant."
)

# ------------------ CHART 2 --------------------------------

st.markdown("## 2. Entry Cost Sensitivity")

cost_values = np.arange(0, 5.01, 0.25)

chart2_df = pd.DataFrame({
    "Entry cost": cost_values,
    "Expected entry payoff": [
        expected_entry_payoff(
            posterior_aggressive,
            c,
            fight_payoff,
            accommodation_payoff
        )
        for c in cost_values
    ]
})

fig2 = px.line(
    chart2_df,
    x="Entry cost",
    y="Expected entry payoff",
    markers=True,
    title="Effect of Fixed Entry Costs"
)

fig2.add_hline(y=0, line_dash="dash")
fig2.update_layout(template="plotly_white")

st.plotly_chart(fig2, use_container_width=True)

st.write(
    "Interpretation: higher fixed entry costs reduce the entrant's "
    "expected payoff, all else being equal."
)

# ------------------ CHART 3 --------------------------------

st.markdown("## 3. Entrant Payoff by Outcome")

payoff_table = pd.DataFrame({
    "Outcome": [
        "Stay out",
        "Enter and incumbent fights",
        "Enter and incumbent accommodates"
    ],
    "Entrant payoff": [
        0,
        fight_payoff - entry_cost,
        accommodation_payoff - entry_cost
    ],
    "Explanation": [
        "No entry investment",
        "Competitive retaliation",
        "Competition without a price war"
    ]
})

fig3 = px.bar(
    payoff_table,
    x="Outcome",
    y="Entrant payoff",
    color="Entrant payoff",
    hover_data=["Explanation"],
    title="Payoff Comparison Across Outcomes"
)

fig3.add_hline(y=0, line_dash="dash")
fig3.update_layout(template="plotly_white")

st.plotly_chart(fig3, use_container_width=True)

st.dataframe(
    payoff_table,
    use_container_width=True,
    hide_index=True
)

# ------------------ CHART 4 --------------------------------

st.markdown("## 4. Decision Heatmap")

heat_probabilities = np.linspace(0, 1, 21)
heat_costs = np.arange(0, 5.01, 0.5)

heatmap_values = np.array([
    [
        expected_entry_payoff(
            p, c, fight_payoff, accommodation_payoff
        )
        for p in heat_probabilities
    ]
    for c in heat_costs
])

fig4 = go.Figure(
    data=go.Heatmap(
        z=heatmap_values,
        x=heat_probabilities,
        y=heat_costs,
        colorscale="RdYlGn",
        colorbar={"title": "Expected payoff"},
        hovertemplate=(
            "Aggression probability: %{x:.0%}<br>"
            "Entry cost: %{y:.2f}<br>"
            "Expected payoff: %{z:.2f}<extra></extra>"
        )
    )
)

fig4.update_layout(
    title="Expected Payoff Across Entry Costs and Aggression Beliefs",
    xaxis_title="Probability of aggressive incumbent",
    yaxis_title="Fixed entry cost",
    template="plotly_white"
)

fig4.update_xaxes(tickformat=".0%")

st.plotly_chart(fig4, use_container_width=True)

st.write(
    "Interpretation: positive expected payoffs favour entry under "
    "this preliminary decision rule; negative payoffs favour staying "
    "out. A zero payoff indicates indifference."
)

# ------------------ CHART 5 --------------------------------

st.markdown("## 5. Bayesian Belief Updating")

belief_df = pd.DataFrame({
    "Belief stage": ["Prior", "Posterior"],
    "Probability of aggressive incumbent": [
        prior_aggressive,
        posterior_aggressive
    ]
})

fig5 = px.bar(
    belief_df,
    x="Belief stage",
    y="Probability of aggressive incumbent",
    text="Probability of aggressive incumbent",
    title="Prior vs. Posterior Probability"
)

fig5.update_traces(
    texttemplate="%{text:.1%}",
    textposition="outside"
)

fig5.update_yaxes(range=[0, 1], tickformat=".0%")
fig5.update_layout(template="plotly_white")

st.plotly_chart(fig5, use_container_width=True)

st.write(
    "The posterior probability is calculated using Bayes' rule. "
    "The signal likelihoods are user-defined assumptions; they "
    "are not observed estimates from an actual incumbent."
)

# ------------------ CHART 6 --------------------------------

st.markdown("## 6. Scenario Comparison")

scenario_definitions = [
    (
        "Optimistic",
        0.20,
        accommodation_payoff + 1
    ),
    (
        "Baseline",
        posterior_aggressive,
        accommodation_payoff
    ),
    (
        "Pessimistic",
        0.80,
        max(0, accommodation_payoff - 2)
    )
]

scenario_rows = []

for name, probability, market_payoff in scenario_definitions:
    payoff = expected_entry_payoff(
        probability,
        entry_cost,
        fight_payoff,
        market_payoff
    )

    scenario_rows.append({
        "Scenario": name,
        "Aggressive probability": probability,
        "Accommodation payoff": market_payoff,
        "Expected entry payoff": payoff,
        "Preliminary decision": decision_rule(payoff)
    })

scenario_df = pd.DataFrame(scenario_rows)

fig6 = px.bar(
    scenario_df,
    x="Scenario",
    y="Expected entry payoff",
    color="Preliminary decision",
    text="Expected entry payoff",
    title="Comparison of Illustrative Scenarios"
)

fig6.add_hline(y=0, line_dash="dash")
fig6.update_traces(texttemplate="%{text:.2f}", textposition="outside")
fig6.update_layout(template="plotly_white")

st.plotly_chart(fig6, use_container_width=True)

st.dataframe(
    scenario_df.round(3),
    use_container_width=True,
    hide_index=True
)

# ------------------ MODEL NOTES ----------------------------

st.divider()

st.markdown("## Model Assumptions and Limitations")

st.markdown(
    """
    - The incumbent is assumed to fight if aggressive and
      accommodate otherwise.
    - The entrant's stay-out payoff is normalized to zero.
    - All payoffs and signal likelihoods are hypothetical.
    - The signal is assumed to have the likelihood specified
      in the sidebar, conditional on the incumbent's type.
    - The incumbent's type-specific optimal response has not
      been solved in this preliminary model.
    - Therefore, the displayed decision is not a claim of
      Bayesian Nash Equilibrium or Perfect Bayesian Equilibrium.
    - Real-world predictions would require suitable data,
      defensible calibration and validation.
    """
)

st.markdown("## Research and Responsible AI")

st.markdown(
    """
    This dashboard demonstrates scenario analysis and Bayesian
    belief updating. AI tools may assist with coding and interpreting
    results, but the assumptions and calculations must be verified.

    The results should be used as decision support rather than as
    an automatic recommendation for a real company.
    """
)

# ------------------ DOWNLOADS ------------------------------

st.divider()
st.markdown("## Download Project Data")

sensitivity_df = pd.DataFrame([
    {
        "Aggressive probability": p,
        "Entry cost": c,
        "Expected entry payoff": expected_entry_payoff(
            p, c, fight_payoff, accommodation_payoff
        ),
        "Decision": decision_rule(
            expected_entry_payoff(
                p, c, fight_payoff, accommodation_payoff
            )
        )
    }
    for p in np.linspace(0, 1, 21)
    for c in np.arange(0, 5.01, 0.5)
])

download1, download2, download3 = st.columns(3)

with download1:
    st.download_button(
        "Download Sensitivity CSV",
        data=sensitivity_df.to_csv(index=False).encode("utf-8"),
        file_name="entry_deterrence_sensitivity.csv",
        mime="text/csv"
    )

with download2:
    st.download_button(
        "Download Scenario CSV",
        data=scenario_df.to_csv(index=False).encode("utf-8"),
        file_name="entry_deterrence_scenarios.csv",
        mime="text/csv"
    )

with download3:
    st.download_button(
        "Download Payoff CSV",
        data=payoff_table.to_csv(index=False).encode("utf-8"),
        file_name="entry_deterrence_payoff_table.csv",
        mime="text/csv"
    )

st.caption(
    "WAI Project | Games of Incomplete Information | "
    "Illustrative academic simulation"
)

