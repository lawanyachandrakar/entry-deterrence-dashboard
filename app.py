
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# ENTRY DETERRENCE GAME
# Improved static Bayesian game with private incumbent type
# All numerical values are illustrative payoff units.
# ==========================================================

st.set_page_config(
    page_title="Entry Deterrence Game",
    page_icon="📊",
    layout="wide"
)

st.title("The Entry Deterrence Game")
st.subheader(
    "AI-Assisted Startup Market Entry Under Incomplete Information"
)

st.markdown(
    """
    This dashboard models a startup deciding whether to enter a market
    when the incumbent's competitive type is not directly observable.
    It calculates incumbent best responses by type and evaluates the
    entrant's expected payoff under its beliefs.
    """
)

st.info(
    "All payoffs and signal likelihoods are illustrative assumptions. "
    "The static model is solved under the specified payoff structure; "
    "it is not a prediction of any actual company's behaviour."
)

# ----------------------------------------------------------
# 1. SIDEBAR: MODEL PARAMETERS
# ----------------------------------------------------------

st.sidebar.header("Game Parameters")

st.sidebar.subheader("A. Entrant's beliefs")

prior_aggressive = st.sidebar.slider(
    "Prior probability of aggressive incumbent",
    min_value=0,
    max_value=100,
    value=40,
    step=5
) / 100

st.sidebar.subheader("B. Entrant payoffs")

entrant_fight_payoff = st.sidebar.number_input(
    "Entrant payoff if incumbent fights (before entry cost)",
    value=-3.0,
    step=0.5
)

entrant_accommodate_payoff = st.sidebar.number_input(
    "Entrant payoff if incumbent accommodates (before entry cost)",
    value=4.0,
    step=0.5
)

entry_cost = st.sidebar.slider(
    "Fixed entry cost",
    min_value=0.0,
    max_value=5.0,
    value=2.0,
    step=0.5
)

stay_out_payoff = st.sidebar.number_input(
    "Entrant payoff from staying out",
    value=0.0,
    step=0.5
)

st.sidebar.subheader("C. Incumbent payoffs")

st.sidebar.caption(
    "Incumbent payoff values are shown before any additional "
    "type-specific costs. Change them to test best responses."
)

aggressive_fight = st.sidebar.number_input(
    "Aggressive type: payoff from fighting",
    value=5.0,
    step=0.5
)

aggressive_accommodate = st.sidebar.number_input(
    "Aggressive type: payoff from accommodating",
    value=2.0,
    step=0.5
)

accommodating_fight = st.sidebar.number_input(
    "Accommodating type: payoff from fighting",
    value=0.0,
    step=0.5
)

accommodating_accommodate = st.sidebar.number_input(
    "Accommodating type: payoff from accommodating",
    value=4.0,
    step=0.5
)

st.sidebar.subheader("D. Optional signal experiment")

signal = st.sidebar.selectbox(
    "Observed signal",
    [
        "No new signal",
        "Warning signal",
        "Reassuring signal"
    ]
)

p_warning_aggressive = st.sidebar.slider(
    "P(warning | aggressive)",
    min_value=0.05,
    max_value=0.95,
    value=0.80,
    step=0.05
)

p_warning_accommodating = st.sidebar.slider(
    "P(warning | accommodating)",
    min_value=0.05,
    max_value=0.95,
    value=0.30,
    step=0.05
)

# ----------------------------------------------------------
# 2. INCUMBENT BEST RESPONSES
# ----------------------------------------------------------

def incumbent_best_response(fight_payoff, accommodate_payoff):
    """
    Select the action that maximizes the incumbent type's payoff.
    If payoffs are equal, both actions are best responses.
    """

    if fight_payoff > accommodate_payoff:
        return "Fight"
    elif accommodate_payoff > fight_payoff:
        return "Accommodate"
    else:
        return "Both actions"


aggressive_response = incumbent_best_response(
    aggressive_fight,
    aggressive_accommodate
)

accommodating_response = incumbent_best_response(
    accommodating_fight,
    accommodating_accommodate
)


def entrant_payoff_for_type(
    response,
    fight_payoff,
    accommodate_payoff,
    cost
):
    """Entrant payoff conditional on the incumbent's type."""

    if response == "Fight":
        return fight_payoff - cost

    if response == "Accommodate":
        return accommodate_payoff - cost

    # If the incumbent is indifferent, its response is not unique.
    # Return the range of possible entrant payoffs.
    return (
        min(fight_payoff, accommodate_payoff) - cost,
        max(fight_payoff, accommodate_payoff) - cost
    )


# ----------------------------------------------------------
# 3. STATIC BAYESIAN GAME SOLUTION
# ----------------------------------------------------------

def calculate_entry_payoff(
    prior,
    aggressive_response,
    accommodating_response,
    fight_payoff,
    accommodate_payoff,
    cost
):
    """
    Calculate entrant expected payoff using the incumbent types'
    best responses.

    If a type is indifferent, return a payoff interval because
    its action is not uniquely determined.
    """

    aggressive_result = entrant_payoff_for_type(
        aggressive_response,
        fight_payoff,
        accommodate_payoff,
        cost
    )

    accommodating_result = entrant_payoff_for_type(
        accommodating_response,
        fight_payoff,
        accommodate_payoff,
        cost
    )

    if (
        isinstance(aggressive_result, tuple)
        or isinstance(accommodating_result, tuple)
    ):
        def bounds(value):
            if isinstance(value, tuple):
                return value
            return (value, value)

        a_low, a_high = bounds(aggressive_result)
        c_low, c_high = bounds(accommodating_result)

        lower = prior * a_low + (1 - prior) * c_low
        upper = prior * a_high + (1 - prior) * c_high

        return lower, upper

    return (
        prior * aggressive_result
        + (1 - prior) * accommodating_result
    )


entry_result = calculate_entry_payoff(
    prior_aggressive,
    aggressive_response,
    accommodating_response,
    entrant_fight_payoff,
    entrant_accommodate_payoff,
    entry_cost
)


def classify_entry_result(result, stay_out):
    """Determine the entrant's preferred action."""

    if isinstance(result, tuple):
        lower, upper = result

        if lower > stay_out:
            return "ENTER"
        if upper < stay_out:
            return "STAY OUT"
        return "MULTIPLE OUTCOMES POSSIBLE"

    if result > stay_out:
        return "ENTER"
    if result < stay_out:
        return "STAY OUT"

    return "INDIFFERENT"


decision = classify_entry_result(entry_result, stay_out_payoff)

if isinstance(entry_result, tuple):
    payoff_display = (
        f"{entry_result[0]:.2f} to {entry_result[1]:.2f}"
    )
else:
    payoff_display = f"{entry_result:.2f}"


# ----------------------------------------------------------
# 4. BAYESIAN BELIEF UPDATE: OPTIONAL EXTENSION
# ----------------------------------------------------------

def posterior_probability(
    prior,
    observed_signal,
    p_warning_aggressive,
    p_warning_accommodating
):
    """Apply Bayes' rule to an observed signal."""

    if observed_signal == "No new signal":
        return prior

    if observed_signal == "Warning signal":
        likelihood_a = p_warning_aggressive
        likelihood_c = p_warning_accommodating
    else:
        likelihood_a = 1 - p_warning_aggressive
        likelihood_c = 1 - p_warning_accommodating

    numerator = prior * likelihood_a
    denominator = (
        prior * likelihood_a
        + (1 - prior) * likelihood_c
    )

    if denominator == 0:
        return prior

    return numerator / denominator


posterior_aggressive = posterior_probability(
    prior_aggressive,
    signal,
    p_warning_aggressive,
    p_warning_accommodating
)

posterior_entry_result = calculate_entry_payoff(
    posterior_aggressive,
    aggressive_response,
    accommodating_response,
    entrant_fight_payoff,
    entrant_accommodate_payoff,
    entry_cost
)

posterior_decision = classify_entry_result(
    posterior_entry_result,
    stay_out_payoff
)


# ----------------------------------------------------------
# 5. DASHBOARD METRICS
# ----------------------------------------------------------

st.header("Executive Dashboard")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Aggressive-type prior",
    f"{prior_aggressive:.0%}"
)

col2.metric(
    "Aggressive-type posterior",
    f"{posterior_aggressive:.1%}",
    delta=f"{posterior_aggressive - prior_aggressive:+.1%}"
)

col3.metric(
    "Entry payoff",
    payoff_display
)

col4.metric(
    "Preferred action",
    decision
)

st.caption(
    "The posterior is used for the optional signal scenario. "
    "The static Bayesian game uses the prior belief; the signal "
    "extension represents a later decision after new information."
)

st.divider()

# ----------------------------------------------------------
# 6. CHART 1: PAYOFF VS. PRIOR PROBABILITY
# ----------------------------------------------------------

st.header("1. Expected Entry Payoff vs. Aggression Probability")

probabilities = np.linspace(0, 1, 101)
chart1_rows = []

for p in probabilities:
    result = calculate_entry_payoff(
        p,
        aggressive_response,
        accommodating_response,
        entrant_fight_payoff,
        entrant_accommodate_payoff,
        entry_cost
    )

    if isinstance(result, tuple):
        low, high = result
        payoff = (low + high) / 2
    else:
        payoff = result

    chart1_rows.append({
        "Aggressive probability": p,
        "Expected entry payoff": payoff
    })

chart1_df = pd.DataFrame(chart1_rows)

fig1 = px.line(
    chart1_df,
    x="Aggressive probability",
    y="Expected entry payoff",
    title="Expected Entry Payoff Across Prior Beliefs"
)

fig1.add_hline(
    y=stay_out_payoff,
    line_dash="dash",
    annotation_text="Stay-out payoff"
)

fig1.add_vline(
    x=prior_aggressive,
    line_dash="dot",
    annotation_text="Current prior"
)

fig1.update_xaxes(tickformat=".0%")
fig1.update_layout(template="plotly_white")

st.plotly_chart(fig1, use_container_width=True)

# ----------------------------------------------------------
# 7. CHART 2: ENTRY COST SENSITIVITY
# ----------------------------------------------------------

st.header("2. Entry Cost Sensitivity")

cost_values = np.linspace(0, 5, 51)
cost_rows = []

for cost in cost_values:
    result = calculate_entry_payoff(
        prior_aggressive,
        aggressive_response,
        accommodating_response,
        entrant_fight_payoff,
        entrant_accommodate_payoff,
        cost
    )

    payoff = (
        sum(result) / 2
        if isinstance(result, tuple)
        else result
    )

    cost_rows.append({
        "Entry cost": cost,
        "Expected entry payoff": payoff
    })

cost_df = pd.DataFrame(cost_rows)

fig2 = px.line(
    cost_df,
    x="Entry cost",
    y="Expected entry payoff",
    title="Effect of Entry Costs on the Entrant's Payoff"
)

fig2.add_hline(y=stay_out_payoff, line_dash="dash")
fig2.update_layout(template="plotly_white")

st.plotly_chart(fig2, use_container_width=True)

# ----------------------------------------------------------
# 8. CHART 3: TYPE-SPECIFIC PAYOFFS AND BEST RESPONSES
# ----------------------------------------------------------

st.header("3. Incumbent Payoffs and Best Responses")

incumbent_df = pd.DataFrame([
    {
        "Incumbent type": "Aggressive",
        "Action": "Fight",
        "Incumbent payoff": aggressive_fight,
        "Entrant payoff before cost": entrant_fight_payoff
    },
    {
        "Incumbent type": "Aggressive",
        "Action": "Accommodate",
        "Incumbent payoff": aggressive_accommodate,
        "Entrant payoff before cost": entrant_accommodate_payoff
    },
    {
        "Incumbent type": "Accommodating",
        "Action": "Fight",
        "Incumbent payoff": accommodating_fight,
        "Entrant payoff before cost": entrant_fight_payoff
    },
    {
        "Incumbent type": "Accommodating",
        "Action": "Accommodate",
        "Incumbent payoff": accommodating_accommodate,
        "Entrant payoff before cost": entrant_accommodate_payoff
    }
])

fig3 = px.bar(
    incumbent_df,
    x="Action",
    y="Incumbent payoff",
    color="Incumbent type",
    barmode="group",
    title="Incumbent Payoffs by Type and Action",
    hover_data=["Entrant payoff before cost"]
)

fig3.update_layout(template="plotly_white")
st.plotly_chart(fig3, use_container_width=True)

st.markdown("**Optimal response by incumbent type**")

response_df = pd.DataFrame([
    {
        "Incumbent type": "Aggressive",
        "Fight payoff": aggressive_fight,
        "Accommodate payoff": aggressive_accommodate,
        "Best response": aggressive_response
    },
    {
        "Incumbent type": "Accommodating",
        "Fight payoff": accommodating_fight,
        "Accommodate payoff": accommodating_accommodate,
        "Best response": accommodating_response
    }
])

st.dataframe(response_df, use_container_width=True, hide_index=True)

# ----------------------------------------------------------
# 9. CHART 4: DECISION HEATMAP
# ----------------------------------------------------------

st.header("4. Entry Decision Heatmap")

heat_probabilities = np.linspace(0, 1, 41)
heat_costs = np.linspace(0, 5, 26)

heat_values = []

for cost in heat_costs:
    row = []

    for p in heat_probabilities:
        result = calculate_entry_payoff(
            p,
            aggressive_response,
            accommodating_response,
            entrant_fight_payoff,
            entrant_accommodate_payoff,
            cost
        )

        payoff = (
            sum(result) / 2
            if isinstance(result, tuple)
            else result
        )

        row.append(payoff)

    heat_values.append(row)

fig4 = go.Figure(
    data=go.Heatmap(
        z=heat_values,
        x=heat_probabilities,
        y=heat_costs,
        colorscale="RdYlGn",
        colorbar={"title": "Entry payoff"},
        hovertemplate=(
            "Aggression probability: %{x:.0%}<br>"
            "Entry cost: %{y:.2f}<br>"
            "Expected payoff: %{z:.2f}<extra></extra>"
        )
    )
)

fig4.update_layout(
    title="Expected Payoff by Belief and Entry Cost",
    xaxis_title="Prior probability of aggressive incumbent",
    yaxis_title="Entry cost",
    template="plotly_white"
)

fig4.update_xaxes(tickformat=".0%")
st.plotly_chart(fig4, use_container_width=True)

# ----------------------------------------------------------
# 10. CHART 5: BELIEF UPDATING
# ----------------------------------------------------------

st.header("5. Bayesian Belief Updating")

belief_df = pd.DataFrame({
    "Belief": ["Prior", "Posterior"],
    "Probability of aggressive type": [
        prior_aggressive,
        posterior_aggressive
    ]
})

fig5 = px.bar(
    belief_df,
    x="Belief",
    y="Probability of aggressive type",
    text="Probability of aggressive type",
    title="Belief Before and After an Observed Signal"
)

fig5.update_traces(
    texttemplate="%{text:.1%}",
    textposition="outside"
)

fig5.update_yaxes(range=[0, 1], tickformat=".0%")
fig5.update_layout(template="plotly_white")

st.plotly_chart(fig5, use_container_width=True)

st.markdown(
    f"""
    After the selected signal, the probability of an aggressive type
    is **{posterior_aggressive:.1%}**. This result depends on the
    assumed signal likelihoods and is not an empirical estimate.
    """
)

# ----------------------------------------------------------
# 11. CHART 6: SCENARIO COMPARISON
# ----------------------------------------------------------

st.header("6. Scenario Comparison")

scenario_specs = [
    ("Optimistic", 0.20, entrant_accommodate_payoff + 1),
    ("Baseline", prior_aggressive, entrant_accommodate_payoff),
    ("Pessimistic", 0.80, max(0, entrant_accommodate_payoff - 2))
]

scenario_rows = []

for name, probability, accommodation_payoff in scenario_specs:
    result = calculate_entry_payoff(
        probability,
        aggressive_response,
        accommodating_response,
        entrant_fight_payoff,
        accommodation_payoff,
        entry_cost
    )

    payoff = (
        sum(result) / 2
        if isinstance(result, tuple)
        else result
    )

    scenario_rows.append({
        "Scenario": name,
        "Aggressive probability": probability,
        "Accommodation payoff": accommodation_payoff,
        "Expected entry payoff": payoff,
        "Preferred action": decision_rule(payoff)
    })

scenario_df = pd.DataFrame(scenario_rows)

fig6 = px.bar(
    scenario_df,
    x="Scenario",
    y="Expected entry payoff",
    color="Preferred action",
    text="Expected entry payoff",
    title="Entry Payoff Under Alternative Scenarios"
)

fig6.add_hline(y=stay_out_payoff, line_dash="dash")
fig6.update_traces(texttemplate="%{text:.2f}", textposition="outside")
fig6.update_layout(template="plotly_white")

st.plotly_chart(fig6, use_container_width=True)
st.dataframe(scenario_df.round(3), use_container_width=True, hide_index=True)

# ----------------------------------------------------------
# 12. EQUILIBRIUM INTERPRETATION
# ----------------------------------------------------------

st.header("Equilibrium Interpretation")

st.markdown(
    f"""
    - Aggressive incumbent best response: **{aggressive_response}**
    - Accommodating incumbent best response: **{accommodating_response}**
    - Entrant's prior probability of the aggressive type: **{prior_aggressive:.0%}**
    - Entrant's expected payoff from entry: **{payoff_display}**
    - Entrant's payoff from staying out: **{stay_out_payoff:.2f}**
    - Entrant's preferred action: **{decision}**
    """
)

st.markdown(
    """
    Under the static model, the incumbent's type-dependent actions
    are best responses to entry. The entrant chooses its action using
    its prior beliefs and the expected payoff implied by those responses.

    When each type has a unique best response and the entrant's
    decision is strict, the resulting strategies form a pure-strategy
    Bayesian Nash Equilibrium of this model.

    If a type is indifferent, or if the entrant is indifferent, multiple
    equilibrium strategies may be possible. The model must then report
    the tie rather than claim a unique equilibrium.
    """
)

# ----------------------------------------------------------
# 13. ASSUMPTIONS AND ETHICS
# ----------------------------------------------------------

st.header("Assumptions, Limitations and Responsible AI")

st.markdown(
    """
    1. All payoff values are illustrative and expressed in normalized
       payoff units.
    2. The entrant cannot observe the incumbent's type before making
       its entry decision.
    3. The incumbent observes its own type and chooses its response
       to maximize its own payoff if entry occurs.
    4. The static model uses prior beliefs. The signal experiment is
       a separate extension in which the entrant receives information.
    5. The model does not establish real-world predictive accuracy.
    6. AI-assisted code and interpretations must be checked against
       the mathematical model.
    7. Any empirical application requires documented, credible data
       and justified parameter estimates.
    """
)

# ----------------------------------------------------------
# 14. DOWNLOAD DATA
# ----------------------------------------------------------

st.header("Download Simulation Results")

sensitivity_rows = []

for p in np.linspace(0, 1, 21):
    for cost in np.arange(0, 5.01, 0.5):
        result = calculate_entry_payoff(
            p,
            aggressive_response,
            accommodating_response,
            entrant_fight_payoff,
            entrant_accommodate_payoff,
            cost
        )

        payoff = (
            sum(result) / 2
            if isinstance(result, tuple)
            else result
        )

        sensitivity_rows.append({
            "Aggressive prior": p,
            "Entry cost": cost,
            "Expected entry payoff": payoff,
            "Preferred action": decision_rule(payoff)
        })

sensitivity_df = pd.DataFrame(sensitivity_rows)

incumbent_payoff_table = pd.DataFrame([
    {
        "Type": "Aggressive",
        "Fight payoff": aggressive_fight,
        "Accommodate payoff": aggressive_accommodate,
        "Best response": aggressive_response
    },
    {
        "Type": "Accommodating",
        "Fight payoff": accommodating_fight,
        "Accommodate payoff": accommodating_accommodate,
        "Best response": accommodating_response
    }
])

c1, c2, c3 = st.columns(3)

with c1:
    st.download_button(
        "Download Sensitivity CSV",
        data=sensitivity_df.to_csv(index=False).encode("utf-8"),
        file_name="entry_deterrence_sensitivity.csv",
        mime="text/csv"
    )

with c2:
    st.download_button(
        "Download Scenarios CSV",
        data=scenario_df.to_csv(index=False).encode("utf-8"),
        file_name="entry_deterrence_scenarios.csv",
        mime="text/csv"
    )

with c3:
    st.download_button(
        "Download Incumbent Payoffs CSV",
        data=incumbent_payoff_table.to_csv(index=False).encode("utf-8"),
        file_name="incumbent_type_payoffs.csv",
        mime="text/csv"
    )

st.caption(
    "Working with AI | Games of Incomplete Information | "
    "Illustrative academic simulation"
)
