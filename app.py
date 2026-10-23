
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# ENTRY DETERRENCE GAME
# WAI Project | Games of Incomplete Information
# All payoff values are illustrative.
# ============================================================

st.set_page_config(
    page_title="Entry Deterrence Game",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# 1. HELPER FUNCTIONS
# ============================================================

def decision_rule(payoff, stay_out=0.0):
    """Compare entry payoff against the stay-out payoff."""
    if payoff > stay_out + 1e-9:
        return "ENTER"
    if payoff < stay_out - 1e-9:
        return "STAY OUT"
    return "INDIFFERENT"


def best_response(fight_payoff, accommodate_payoff):
    """Find an incumbent type's payoff-maximizing response."""
    if fight_payoff > accommodate_payoff + 1e-9:
        return "Fight"
    if accommodate_payoff > fight_payoff + 1e-9:
        return "Accommodate"
    return "Both actions"


def bayesian_posterior(
    prior,
    signal,
    p_warning_aggressive,
    p_warning_accommodating
):
    """Calculate the posterior probability of an aggressive type."""
    if signal == "No new signal":
        return prior

    if signal == "Warning signal":
        likelihood_a = p_warning_aggressive
        likelihood_c = p_warning_accommodating
    else:
        likelihood_a = 1 - p_warning_aggressive
        likelihood_c = 1 - p_warning_accommodating

    numerator = prior * likelihood_a
    denominator = (
        numerator + (1 - prior) * likelihood_c
    )

    if denominator <= 0:
        return prior

    return numerator / denominator


def entrant_payoff_for_type(
    response,
    fight_payoff,
    accommodate_payoff,
    entry_cost
):
    """Entrant payoff conditional on an incumbent's response."""
    if response == "Fight":
        return fight_payoff - entry_cost

    if response == "Accommodate":
        return accommodate_payoff - entry_cost

    # The incumbent is indifferent between both actions.
    values = [
        fight_payoff - entry_cost,
        accommodate_payoff - entry_cost
    ]
    return min(values), max(values)


def expected_entry_payoff(
    probability_aggressive,
    response_aggressive,
    response_accommodating,
    fight_payoff,
    accommodate_payoff,
    entry_cost
):
    """
    Calculate the entrant's expected payoff using type-specific
    incumbent best responses.

    Returns a number if responses are unique.
    Returns an interval if either type is indifferent.
    """

    payoff_a = entrant_payoff_for_type(
        response_aggressive,
        fight_payoff,
        accommodate_payoff,
        entry_cost
    )

    # CORRECTED: all arguments passed in the proper order.
    payoff_c = entrant_payoff_for_type(
        response_accommodating,
        fight_payoff,
        accommodate_payoff,
        entry_cost
    )

    def bounds(value):
        if isinstance(value, tuple):
            return value
        return value, value

    a_low, a_high = bounds(payoff_a)
    c_low, c_high = bounds(payoff_c)

    low = (
        probability_aggressive * a_low
        + (1 - probability_aggressive) * c_low
    )

    high = (
        probability_aggressive * a_high
        + (1 - probability_aggressive) * c_high
    )

    if abs(high - low) < 1e-9:
        return low

    return low, high


def payoff_midpoint(result):
    """Midpoint for visualizing an interval; not an equilibrium."""
    if isinstance(result, tuple):
        return (result[0] + result[1]) / 2
    return result


def payoff_label(result):
    """Format a payoff or payoff interval."""
    if isinstance(result, tuple):
        return f"{result[0]:.2f} to {result[1]:.2f}"
    return f"{result:.2f}"


def payoff_decision(result, stay_out):
    """Make a decision while respecting payoff intervals."""
    if isinstance(result, tuple):
        low, high = result

        if low > stay_out + 1e-9:
            return "ENTER"
        if high < stay_out - 1e-9:
            return "STAY OUT"
        return "MULTIPLE OUTCOMES POSSIBLE"

    return decision_rule(result, stay_out)


# ============================================================
# 2. DASHBOARD TITLE
# ============================================================

st.title("The Entry Deterrence Game")

st.subheader(
    "AI-Assisted Startup Market Entry Under Incomplete Information"
)

st.markdown(
    """
    This dashboard studies how a startup's market-entry decision
    depends on uncertainty about an incumbent's competitive type.

    Explore type-specific incumbent payoffs, expected entry payoffs,
    Bayesian belief updating, sensitivity analysis and scenarios.
    """
)

st.info(
    "All payoff values and signal probabilities are hypothetical. "
    "They are not empirical estimates of real companies."
)

# ============================================================
# 3. SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("Game Parameters")

st.sidebar.subheader("A. Entrant's beliefs")

prior_aggressive = st.sidebar.slider(
    "Prior probability of aggressive incumbent",
    min_value=0,
    max_value=100,
    value=40,
    step=5
) / 100

st.sidebar.subheader("B. Entrant's payoffs")

entrant_fight = st.sidebar.number_input(
    "Entrant payoff if incumbent fights",
    value=-3.0,
    step=0.5
)

entrant_accommodate = st.sidebar.number_input(
    "Entrant payoff if incumbent accommodates",
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
    "Payoff from staying out",
    value=0.0,
    step=0.5
)

st.sidebar.subheader("C. Incumbent's type-specific payoffs")

st.sidebar.caption(
    "Each type selects the action that maximizes its own payoff."
)

aggressive_fight = st.sidebar.number_input(
    "Aggressive type: fight payoff",
    value=5.0,
    step=0.5
)

aggressive_accommodate = st.sidebar.number_input(
    "Aggressive type: accommodate payoff",
    value=2.0,
    step=0.5
)

accommodating_fight = st.sidebar.number_input(
    "Accommodating type: fight payoff",
    value=0.0,
    step=0.5
)

accommodating_accommodate = st.sidebar.number_input(
    "Accommodating type: accommodate payoff",
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
    "P(warning | aggressive type)",
    min_value=0.05,
    max_value=0.95,
    value=0.80,
    step=0.05
)

p_warning_accommodating = st.sidebar.slider(
    "P(warning | accommodating type)",
    min_value=0.05,
    max_value=0.95,
    value=0.30,
    step=0.05
)

# ============================================================
# 4. SOLVE THE STATIC GAME
# ============================================================

response_a = best_response(
    aggressive_fight,
    aggressive_accommodate
)

response_c = best_response(
    accommodating_fight,
    accommodating_accommodate
)

entry_result = expected_entry_payoff(
    prior_aggressive,
    response_a,
    response_c,
    entrant_fight,
    entrant_accommodate,
    entry_cost
)

decision = payoff_decision(
    entry_result,
    stay_out_payoff
)

# ============================================================
# 5. BAYESIAN BELIEF UPDATING
# ============================================================

posterior_aggressive = bayesian_posterior(
    prior_aggressive,
    signal,
    p_warning_aggressive,
    p_warning_accommodating
)

posterior_result = expected_entry_payoff(
    posterior_aggressive,
    response_a,
    response_c,
    entrant_fight,
    entrant_accommodate,
    entry_cost
)

posterior_decision = payoff_decision(
    posterior_result,
    stay_out_payoff
)

# ============================================================
# 6. EXECUTIVE DASHBOARD
# ============================================================

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
    "Expected entry payoff",
    payoff_label(entry_result)
)

col4.metric(
    "Preferred action",
    decision
)

st.caption(
    "The static game uses prior beliefs. The posterior represents "
    "an optional decision scenario after observing a signal."
)

st.divider()

# ============================================================
# 7. CHART 1: EXPECTED PAYOFF VS. AGGRESSION PROBABILITY
# ============================================================

st.header("1. Expected Payoff vs. Aggression Probability")

probabilities = np.linspace(0, 1, 101)
chart1_rows = []

for p in probabilities:
    result = expected_entry_payoff(
        p,
        response_a,
        response_c,
        entrant_fight,
        entrant_accommodate,
        entry_cost
    )

    chart1_rows.append({
        "Aggressive probability": p,
        "Expected entry payoff": payoff_midpoint(result)
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

st.caption(
    "If an incumbent type is indifferent between actions, the chart "
    "plots the midpoint of the resulting payoff interval for display."
)

# ============================================================
# 8. CHART 2: ENTRY COST SENSITIVITY
# ============================================================

st.header("2. Entry Cost Sensitivity")

cost_values = np.linspace(0, 5, 51)
cost_rows = []

for cost in cost_values:
    result = expected_entry_payoff(
        prior_aggressive,
        response_a,
        response_c,
        entrant_fight,
        entrant_accommodate,
        cost
    )

    cost_rows.append({
        "Entry cost": cost,
        "Expected entry payoff": payoff_midpoint(result)
    })

cost_df = pd.DataFrame(cost_rows)

fig2 = px.line(
    cost_df,
    x="Entry cost",
    y="Expected entry payoff",
    title="Effect of Entry Costs on Expected Payoff"
)

fig2.add_hline(
    y=stay_out_payoff,
    line_dash="dash"
)

fig2.update_layout(template="plotly_white")

st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# 9. CHART 3: INCUMBENT PAYOFFS AND BEST RESPONSES
# ============================================================

st.header("3. Incumbent Payoffs and Best Responses")

incumbent_df = pd.DataFrame([
    {
        "Incumbent type": "Aggressive",
        "Action": "Fight",
        "Incumbent payoff": aggressive_fight
    },
    {
        "Incumbent type": "Aggressive",
        "Action": "Accommodate",
        "Incumbent payoff": aggressive_accommodate
    },
    {
        "Incumbent type": "Accommodating",
        "Action": "Fight",
        "Incumbent payoff": accommodating_fight
    },
    {
        "Incumbent type": "Accommodating",
        "Action": "Accommodate",
        "Incumbent payoff": accommodating_accommodate
    }
])

fig3 = px.bar(
    incumbent_df,
    x="Action",
    y="Incumbent payoff",
    color="Incumbent type",
    barmode="group",
    title="Incumbent Payoffs by Type and Action"
)

fig3.update_layout(template="plotly_white")

st.plotly_chart(fig3, use_container_width=True)

response_df = pd.DataFrame([
    {
        "Incumbent type": "Aggressive",
        "Fight payoff": aggressive_fight,
        "Accommodate payoff": aggressive_accommodate,
        "Best response": response_a
    },
    {
        "Incumbent type": "Accommodating",
        "Fight payoff": accommodating_fight,
        "Accommodate payoff": accommodating_accommodate,
        "Best response": response_c
    }
])

st.markdown("**Incumbent best-response table**")

st.dataframe(
    response_df,
    use_container_width=True,
    hide_index=True
)

# ============================================================
# 10. CHART 4: DECISION HEATMAP
# ============================================================

st.header("4. Entry Decision Heatmap")

heat_probabilities = np.linspace(0, 1, 41)
heat_costs = np.linspace(0, 5, 26)

heat_values = []

for cost in heat_costs:
    row = []

    for p in heat_probabilities:
        result = expected_entry_payoff(
            p,
            response_a,
            response_c,
            entrant_fight,
            entrant_accommodate,
            cost
        )

        row.append(payoff_midpoint(result))

    heat_values.append(row)

fig4 = go.Figure(
    data=go.Heatmap(
        z=heat_values,
        x=heat_probabilities,
        y=heat_costs,
        colorscale="RdYlGn",
        colorbar={"title": "Entry payoff"},
        hovertemplate=(
            "Aggressive probability: %{x:.0%}<br>"
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

# ============================================================
# 11. CHART 5: BAYESIAN BELIEF UPDATING
# ============================================================

st.header("5. Bayesian Belief Updating")

belief_df = pd.DataFrame({
    "Belief stage": ["Prior", "Posterior"],
    "Probability of aggressive type": [
        prior_aggressive,
        posterior_aggressive
    ]
})

fig5 = px.bar(
    belief_df,
    x="Belief stage",
    y="Probability of aggressive type",
    text="Probability of aggressive type",
    title="Prior vs. Posterior Belief"
)

fig5.update_traces(
    texttemplate="%{text:.1%}",
    textposition="outside"
)

fig5.update_yaxes(
    range=[0, 1],
    tickformat=".0%"
)

fig5.update_layout(template="plotly_white")

st.plotly_chart(fig5, use_container_width=True)

st.markdown(
    f"""
    After the selected signal, the posterior probability of an
    aggressive incumbent is **{posterior_aggressive:.1%}**.

    Under this updated belief, the preliminary preferred action is
    **{posterior_decision}**.
    """
)

# ============================================================
# 12. CHART 6: SCENARIO COMPARISON
# ============================================================

st.header("6. Scenario Comparison")

scenario_specs = [
    ("Optimistic", 0.20, entrant_accommodate + 1),
    ("Baseline", prior_aggressive, entrant_accommodate),
    ("Pessimistic", 0.80, max(0, entrant_accommodate - 2))
]

scenario_rows = []

for name, probability, market_payoff in scenario_specs:
    # CORRECTED: the same valid payoff function is used for every
    # scenario, and the incumbent's best responses are retained.
    result = expected_entry_payoff(
        probability,
        response_a,
        response_c,
        entrant_fight,
        market_payoff,
        entry_cost
    )

    scenario_rows.append({
        "Scenario": name,
        "Aggressive probability": probability,
        "Accommodation payoff": market_payoff,
        "Expected entry payoff": payoff_midpoint(result),
        "Payoff range": payoff_label(result),
        "Preferred action": payoff_decision(
            result,
            stay_out_payoff
        )
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

fig6.add_hline(
    y=stay_out_payoff,
    line_dash="dash"
)

fig6.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside"
)

fig6.update_layout(template="plotly_white")

st.plotly_chart(fig6, use_container_width=True)

st.markdown("**Scenario results**")

st.dataframe(
    scenario_df.round(3),
    use_container_width=True,
    hide_index=True
)

# ============================================================
# 13. EQUILIBRIUM INTERPRETATION
# ============================================================

st.divider()

st.header("Equilibrium Interpretation")

st.markdown(
    f"""
    - Aggressive incumbent's best response: **{response_a}**
    - Accommodating incumbent's best response: **{response_c}**
    - Entrant's prior probability of aggression: **{prior_aggressive:.0%}**
    - Expected entry payoff: **{payoff_label(entry_result)}**
    - Stay-out payoff: **{stay_out_payoff:.2f}**
    - Preferred entrant action: **{decision}**
    """
)

st.markdown(
    """
    The incumbent's type-specific actions are selected to maximize
    its payoff if entry occurs. The entrant then compares the expected
    payoff from entry with its payoff from staying out.

    If each incumbent type has a unique best response and the entrant
    has a strict preferred action, the resulting strategy profile
    constitutes a pure-strategy Bayesian Nash Equilibrium of this
    specified static game.

    If a type is indifferent between actions, the entrant's payoff
    may be an interval. The model reports this ambiguity rather than
    automatically claiming a unique equilibrium.

    The optional signal experiment illustrates belief updating. It
    does not by itself solve a dynamic Perfect Bayesian Equilibrium.
    """
)

# ============================================================
# 14. ASSUMPTIONS AND RESPONSIBLE AI
# ============================================================

st.header("Assumptions, Limitations and Responsible AI")

st.markdown(
    """
    1. All payoffs are hypothetical normalized units.
    2. The entrant does not observe the incumbent's type before entry.
    3. The incumbent knows its own type and chooses its best response.
    4. The static game uses prior beliefs; the signal extension
       illustrates a separate decision after new information.
    5. No real-world predictive accuracy is claimed.
    6. AI-assisted code and interpretations must be verified.
    7. Real-world application requires credible data and justified
       payoff assumptions.
    """
)

# ============================================================
# 15. DOWNLOAD RESULTS
# ============================================================

st.divider()

st.header("Download Simulation Results")

sensitivity_rows = []

for p in np.linspace(0, 1, 21):
    for cost in np.arange(0, 5.01, 0.5):
        result = expected_entry_payoff(
            p,
            response_a,
            response_c,
            entrant_fight,
            entrant_accommodate,
            cost
        )

        sensitivity_rows.append({
            "Aggressive prior": p,
            "Entry cost": cost,
            "Expected entry payoff": payoff_midpoint(result),
            "Payoff range": payoff_label(result),
            "Preferred action": payoff_decision(
                result,
                stay_out_payoff
            )
        })

sensitivity_df = pd.DataFrame(sensitivity_rows)

incumbent_payoff_table = pd.DataFrame([
    {
        "Incumbent type": "Aggressive",
        "Fight payoff": aggressive_fight,
        "Accommodate payoff": aggressive_accommodate,
        "Best response": response_a
    },
    {
        "Incumbent type": "Accommodating",
        "Fight payoff": accommodating_fight,
        "Accommodate payoff": accommodating_accommodate,
        "Best response": response_c
    }
])

col1, col2, col3 = st.columns(3)

with col1:
    st.download_button(
        label="Download Sensitivity CSV",
        data=sensitivity_df.to_csv(index=False).encode("utf-8"),
        file_name="entry_deterrence_sensitivity.csv",
        mime="text/csv"
    )

with col2:
    st.download_button(
        label="Download Scenarios CSV",
        data=scenario_df.to_csv(index=False).encode("utf-8"),
        file_name="entry_deterrence_scenarios.csv",
        mime="text/csv"
    )

with col3:
    st.download_button(
        label="Download Incumbent Payoffs CSV",
        data=incumbent_payoff_table.to_csv(index=False).encode("utf-8"),
        file_name="incumbent_type_payoffs.csv",
        mime="text/csv"
    )

st.caption(
    "WAI Project | Games of Incomplete Information | "
    "Illustrative academic simulation"
)
