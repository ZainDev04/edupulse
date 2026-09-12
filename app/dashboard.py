"""EduPulse interactive dashboard (Streamlit).

Run with::

    edupulse dashboard        # or: streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from edupulse import __version__
from edupulse.config import get_settings
from edupulse.data.schema import ETHNICITIES, GENDERS, LUNCH, PARENTAL_EDUCATION, SCORE_COLUMNS, TEST_PREP
from edupulse.models.registry import ModelRegistry
from edupulse.pipeline import load_engineered
from edupulse.serving import PredictionService
from edupulse.tasks import TASKS

st.set_page_config(page_title="EduPulse", page_icon=":mortar_board:", layout="wide", initial_sidebar_state="expanded")
SETTINGS = get_settings()
ACCENT = "#4F46E5"

st.markdown(
    """
    <style>
      .metric-card {background: linear-gradient(135deg,#4F46E5 0%,#7C3AED 100%); color:white; padding:18px 22px;
                    border-radius:14px; box-shadow:0 6px 18px rgba(79,70,229,.25);}
      .metric-card h3 {margin:0; font-size:0.85rem; opacity:.85; font-weight:500; letter-spacing:.04em}
      .metric-card p {margin:4px 0 0 0; font-size:1.9rem; font-weight:700}
      .band-low {color:#10B981;font-weight:700} .band-moderate{color:#F59E0B;font-weight:700}
      .band-high {color:#F97316;font-weight:700} .band-critical{color:#EF4444;font-weight:700}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def data() -> pd.DataFrame:
    return load_engineered(SETTINGS)


@st.cache_resource(show_spinner=False)
def service() -> PredictionService:
    return PredictionService(SETTINGS.models_dir)


@st.cache_data(show_spinner=False)
def summary() -> dict:
    p = SETTINGS.reports_dir / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def card(col, label: str, value: str):
    col.markdown(f'<div class="metric-card"><h3>{label}</h3><p>{value}</p></div>', unsafe_allow_html=True)


def trained_tasks() -> list[str]:
    return service().available_tasks()


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_overview():
    df = data()
    st.title("EduPulse: student performance intelligence")
    st.caption(f"v{__version__}, {len(df):,} students, 3 prediction tasks, SHAP explanations, fairness audit")

    c1, c2, c3, c4 = st.columns(4)
    card(c1, "STUDENTS", f"{len(df):,}")
    card(c2, "AT-RISK RATE", f"{df['at_risk'].mean():.1%}")
    card(c3, "AVG SCORE", f"{df['average_score'].mean():.1f}")
    s = summary()
    card(c4, "AT-RISK MODEL AUC", f"{s['at_risk']['test_metrics']['roc_auc']:.3f}" if "at_risk" in s else "n/a")

    st.markdown("### What this platform does")
    st.markdown(
        """
        | Task | Question it answers | Inputs |
        |---|---|---|
        | At-risk early warning | Which students are likely to average below 60, before any exam is taken? | background only |
        | Math-score prediction | What math score should we expect, given reading and writing scores and background? | background + reading/writing |
        | Performance tiering | Low, medium or high tier from all scores (the original coursework task) | background + all scores |
        """
    )

    st.markdown("### Explore the data")
    tab1, tab2, tab3 = st.tabs(["Distributions", "Background effects", "Raw data"])
    with tab1:
        col = st.selectbox("Score", SCORE_COLUMNS + ["average_score"], index=3)
        hue = st.selectbox(
            "Colour by", ["test_preparation_course", "lunch", "gender", "parental_level_of_education", "race_ethnicity"]
        )
        fig = px.histogram(
            df,
            x=col,
            color=hue,
            nbins=30,
            barmode="overlay",
            opacity=0.7,
            marginal="box",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        st.plotly_chart(fig, width="stretch")
    with tab2:
        attr = st.selectbox(
            "Attribute", ["lunch", "test_preparation_course", "parental_level_of_education", "race_ethnicity", "gender"]
        )
        g = (
            df.groupby(attr, observed=True)
            .agg(students=("at_risk", "size"), at_risk_rate=("at_risk", "mean"), avg_score=("average_score", "mean"))
            .reset_index()
        )
        fig = px.bar(
            g,
            x=attr,
            y="at_risk_rate",
            text=g["at_risk_rate"].map("{:.1%}".format),
            color="avg_score",
            color_continuous_scale="RdYlGn",
            hover_data=["students"],
        )
        fig.add_hline(y=df["at_risk"].mean(), line_dash="dash", annotation_text="overall")
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(g.style.format({"at_risk_rate": "{:.1%}", "avg_score": "{:.1f}"}), width="stretch")
    with tab3:
        st.dataframe(df, width="stretch", height=420)
        st.download_button(
            "Download engineered dataset (CSV)", df.to_csv(index=False).encode(), "students_engineered.csv", "text/csv"
        )


def page_leaderboard():
    st.title("Model leaderboard")
    tasks = trained_tasks()
    if not tasks:
        st.warning("No trained models found. Run `edupulse train --all` first.")
        return
    task = st.selectbox("Task", tasks, format_func=lambda t: f"{t}: {TASKS[t].description[:70]}...")
    m = service().model(task)
    md = m.metadata
    if m.task.leakage_note:
        st.warning("Leakage note: " + m.task.leakage_note)

    c1, c2, c3, c4 = st.columns(4)
    card(c1, "SELECTED MODEL", md.model_name.replace("_", " "))
    card(c2, f"CV {md.cv_metric.upper()}", f"{md.cv_score:.3f}")
    hm = md.test_metrics
    if m.task.kind == "binary":
        card(c3, "HOLD-OUT ROC-AUC", f"{hm['roc_auc']:.3f}")
        card(c4, "RECALL @ THRESHOLD", f"{hm['recall']:.1%}")
    elif m.task.kind == "regression":
        card(c3, "HOLD-OUT R²", f"{hm['r2']:.3f}")
        card(c4, "RMSE", f"{hm['rmse']:.2f}")
    else:
        card(c3, "HOLD-OUT ACCURACY", f"{hm['accuracy']:.3f}")
        card(c4, "MACRO-F1", f"{hm['f1_macro']:.3f}")

    board = pd.read_csv(m.path / "leaderboard.csv")
    metric = f"{m.task.primary_metric}_mean"
    fig = px.bar(
        board.sort_values(metric),
        x=metric,
        y="model",
        orientation="h",
        error_x=f"{m.task.primary_metric}_std",
        color="family",
        text=board.sort_values(metric)[metric].map("{:.3f}".format),
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_layout(height=480, xaxis_title=f"CV {m.task.primary_metric} (mean ± std)", yaxis_title="")
    st.plotly_chart(fig, width="stretch")

    with st.expander("Full leaderboard table"):
        st.dataframe(board, width="stretch")
    if md.best_params:
        with st.expander("Optuna-tuned hyper-parameters"):
            st.json(md.best_params)

    st.markdown("### Hold-out diagnostics")
    figs = {
        k: v
        for k, v in md.figures.items()
        if Path(v).exists() and k not in ("leaderboard", "permutation", "shap_bar", "shap_beeswarm", "fairness")
    }
    cols = st.columns(2)
    for i, (k, v) in enumerate(figs.items()):
        cols[i % 2].image(v, caption=k.replace("_", " "), width="stretch")


def page_predict():
    st.title("Predict")
    tasks = trained_tasks()
    if not tasks:
        st.warning("No trained models found. Run `edupulse train --all` first.")
        return
    task = st.radio("Task", tasks, horizontal=True, format_func=lambda t: t.replace("_", " ").title())
    t = TASKS[task]

    with st.form("predict"):
        c1, c2, c3 = st.columns(3)
        gender = c1.selectbox("Gender", GENDERS)
        eth = c2.selectbox("Race / ethnicity", ETHNICITIES, index=2)
        edu = c3.selectbox("Parental level of education", PARENTAL_EDUCATION, index=2)
        c4, c5 = st.columns(2)
        lunch = c4.selectbox("Lunch", LUNCH, index=1)
        prep = c5.selectbox("Test preparation course", TEST_PREP)
        row = {
            "gender": gender,
            "race_ethnicity": eth,
            "parental_level_of_education": edu,
            "lunch": lunch,
            "test_preparation_course": prep,
        }
        needed = [s for s in SCORE_COLUMNS if s in t.features]
        if needed:
            cols = st.columns(len(needed))
            for c, s in zip(cols, needed, strict=False):
                row[s] = c.slider(s.replace("_", " ").title(), 0, 100, 65)
        explain = st.checkbox("Explain with SHAP", value=True)
        submitted = st.form_submit_button("Predict", type="primary", width="stretch")

    if not submitted:
        return
    with st.spinner("Scoring..."):
        res = service().predict(task, [row], explain=explain)[0]

    if t.kind == "binary":
        p = res["probability"]
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=100 * p,
                number={"suffix": "%"},
                title={"text": "Probability of averaging < 60"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": ACCENT},
                    "steps": [
                        {"range": [0, 35], "color": "#D1FAE5"},
                        {"range": [35, 60], "color": "#FEF3C7"},
                        {"range": [60, 80], "color": "#FFEDD5"},
                        {"range": [80, 100], "color": "#FEE2E2"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 3}, "value": 100 * res["threshold"]},
                },
            )
        )
        fig.update_layout(height=320, margin={"t": 60, "b": 10})
        c1, c2 = st.columns([2, 1])
        c1.plotly_chart(fig, width="stretch")
        c2.markdown(
            f"### Risk band: <span class='band-{res['risk_band']}'>{res['risk_band'].upper()}</span>",
            unsafe_allow_html=True,
        )
        c2.markdown(
            f"Flag for intervention: {'yes' if res['at_risk'] else 'no'}  \nDecision threshold: {res['threshold']:.2f}  \nModel version: `{res['model_version']}`"
        )
        if res.get("mitigated"):
            m = res["mitigated"]
            c2.markdown(
                f"With the {m['attribute']}-equalised threshold ({m['threshold']:.2f}): "
                f"{'flagged' if m['at_risk'] else 'not flagged'}"
            )
        c2.info("This is a triage signal for prioritising support. It is not a judgement about the student.")
    elif t.kind == "regression":
        st.metric("Predicted math score", f"{res['prediction']:.1f} / 100")
        if res.get("lower") is not None:
            st.caption(
                f"{res['confidence']:.0%} conformal interval: {res['lower']:.1f} to {res['upper']:.1f} "
                "(calibrated on out-of-fold residuals; coverage is a marginal guarantee, not per student)."
            )
    else:
        st.metric("Predicted performance level", res["label"].upper())
        probs = pd.Series(res["probabilities"]).reindex(["low", "medium", "high"])
        st.plotly_chart(
            px.bar(
                x=probs.index,
                y=probs.values,
                labels={"x": "level", "y": "probability"},
                color=probs.index,
                color_discrete_sequence=["#EF4444", "#F59E0B", "#10B981"],
            ),
            width="stretch",
        )

    if explain and res.get("explanation"):
        st.markdown("#### SHAP contributions")
        ex = pd.DataFrame(res["explanation"])
        ex["direction"] = ex["contribution"].apply(
            lambda v: "increases risk / score" if v > 0 else "decreases risk / score"
        )
        fig = px.bar(
            ex.sort_values("contribution"),
            x="contribution",
            y="feature",
            orientation="h",
            color="direction",
            color_discrete_map={"increases risk / score": "#EF4444", "decreases risk / score": "#10B981"},
            hover_data=["value"],
        )
        fig.update_layout(height=340, yaxis_title="")
        st.plotly_chart(fig, width="stretch")


def page_explain():
    st.title("Explainability")
    tasks = trained_tasks()
    if not tasks:
        st.warning("No trained models found.")
        return
    task = st.selectbox("Task", tasks)
    md = service().model(task).metadata
    ex = md.explainability
    c1, c2 = st.columns(2)
    if ex.get("shap_importance"):
        imp = pd.DataFrame(ex["shap_importance"])
        fig = px.bar(
            imp.sort_values("mean_abs_shap"),
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            title="Global importance (mean absolute SHAP)",
            color_discrete_sequence=[ACCENT],
        )
        c1.plotly_chart(fig, width="stretch")
    if ex.get("permutation_importance"):
        perm = pd.DataFrame(ex["permutation_importance"])
        fig = px.bar(
            perm.sort_values("importance_mean"),
            x="importance_mean",
            y="feature",
            orientation="h",
            error_x="importance_std",
            title="Permutation importance (hold-out)",
            color_discrete_sequence=["#F59E0B"],
        )
        c2.plotly_chart(fig, width="stretch")
    bee = md.figures.get("shap_beeswarm")
    if bee and Path(bee).exists():
        st.image(bee, caption="SHAP beeswarm. Each dot is a student; colour is the feature value.", width="stretch")


def page_fairness():
    st.title("Fairness audit")
    tasks = trained_tasks()
    if not tasks:
        st.warning("No trained models found.")
        return
    task = st.selectbox("Task", tasks)
    fair = service().model(task).metadata.fairness
    if not fair.get("groups"):
        st.info("No fairness data for this task.")
        return
    groups = pd.DataFrame(fair["groups"])
    attr = st.selectbox("Sensitive attribute", groups["attribute"].unique())
    g = groups[groups["attribute"] == attr]
    metrics = [c for c in ("selection_rate", "tpr", "fpr", "precision", "roc_auc", "mae", "accuracy") if c in g.columns]
    long = g.melt(id_vars=["group", "n"], value_vars=metrics, var_name="metric", value_name="value").dropna()
    fig = px.bar(
        long,
        x="group",
        y="value",
        color="metric",
        barmode="group",
        hover_data=["n"],
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    st.plotly_chart(fig, width="stretch")
    st.markdown("#### Gap summary (max minus min across groups)")
    st.json(fair.get("summary", {}).get(attr, {}))
    mit = fair.get("mitigated")
    if mit and mit["attribute"] == attr:
        st.markdown(f"#### Mitigation: recall equalised across {attr}")
        before = {r["group"]: r for r in fair["groups"] if r["attribute"] == attr}
        rows = [
            {
                "group": r["group"],
                "threshold": mit["thresholds"].get(r["group"]),
                "recall before": before.get(r["group"], {}).get("tpr"),
                "recall after": r["tpr"],
                "selection before": before.get(r["group"], {}).get("selection_rate"),
                "selection after": r["selection_rate"],
            }
            for r in mit["groups"]
            if r["attribute"] == attr
        ]
        st.dataframe(pd.DataFrame(rows).set_index("group").style.format("{:.3f}"), width="stretch")
        b, a = mit["overall_before"], mit["overall_after"]
        st.caption(
            f"Overall recall {b['recall']:.3f} to {a['recall']:.3f}, precision {b['precision']:.3f} to {a['precision']:.3f}, "
            f"flagged {b['flagged_rate']:.1%} to {a['flagged_rate']:.1%}. Per-group thresholds are chosen on "
            "out-of-fold probabilities so that every group reaches the target recall; the model is unchanged."
        )
    st.caption(
        "Selection-rate and TPR gaps quantify demographic-parity and equal-opportunity differences. "
        "A disparate-impact ratio below 0.8 is the conventional 'four-fifths rule' warning level. "
        "Note that some disparity is *expected* here because the base rates genuinely differ between groups."
    )


def page_monitoring():
    st.title("Drift monitoring")
    tasks = trained_tasks()
    if not tasks:
        st.warning("No trained models found.")
        return
    task = st.selectbox("Task", tasks)
    st.markdown(
        "Upload a CSV of new students (same columns as the training data) to compare with the training "
        "population: population stability index per input and a Kolmogorov-Smirnov test on the model output. "
        "PSI below 0.10 is stable, 0.10 to 0.25 is worth a look, above 0.25 is a material shift."
    )
    upload = st.file_uploader("New cohort CSV", type="csv")
    if upload is None:
        st.info(
            "Waiting for a CSV. The REST API also reports drift on its live prediction stream at /monitoring/drift/{task}."
        )
        return
    from edupulse.data.loader import load_clean

    current = load_clean(upload)
    report = service().drift(task, current)
    if report["status"] == "insufficient":
        st.warning(f"Need at least {report['min_rows']} rows; got {report['n_current']}.")
        return
    colour = {"ok": "green", "warn": "orange", "alert": "red"}[report["status"]]
    st.markdown(
        f"### Overall: :{colour}[{report['status'].upper()}]  "
        f"({report['n_current']} rows against {report['n_reference']} training rows)"
    )
    rows = [{"feature": f["feature"], "psi": f["psi"], "status": f["status"]} for f in report["features"]]
    if report["scores"]:
        rows.append({"feature": "model output", "psi": report["scores"]["psi"], "status": report["scores"]["status"]})
    frame = pd.DataFrame(rows)
    fig = px.bar(
        frame,
        x="psi",
        y="feature",
        orientation="h",
        color="status",
        color_discrete_map={"ok": "#4F46E5", "warn": "#F59E0B", "alert": "#EF4444"},
    )
    fig.add_vline(x=0.25, line_dash="dash", line_color="#94A3B8")
    st.plotly_chart(fig, width="stretch")
    if report["scores"]:
        s = report["scores"]
        st.caption(
            f"Model output: KS statistic {s['ks_statistic']:.3f} (p = {s['ks_pvalue']:.3g}), "
            f"mean {s['reference_mean']:.3f} in training against {s['current_mean']:.3f} now."
        )
    shifts = [
        {"feature": f["feature"], **{k: v for k, v in f["top_shifts"][0].items()}}
        for f in report["features"]
        if f["top_shifts"]
    ]
    st.markdown("#### Largest moving bin per feature")
    st.dataframe(
        pd.DataFrame(shifts).set_index("feature").style.format({"reference": "{:.1%}", "current": "{:.1%}"}),
        width="stretch",
    )


def page_model_card():
    st.title("Model cards")
    tasks = trained_tasks()
    if not tasks:
        st.warning("No trained models found.")
        return
    task = st.selectbox("Task", tasks)
    m = service().model(task)
    st.markdown((m.path / "model_card.md").read_text(encoding="utf-8"))


PAGES = {
    "Overview": page_overview,
    "Leaderboard": page_leaderboard,
    "Predict": page_predict,
    "Explainability": page_explain,
    "Fairness": page_fairness,
    "Monitoring": page_monitoring,
    "Model cards": page_model_card,
}

with st.sidebar:
    st.markdown("## EduPulse")
    choice = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.markdown("---")
    reg = ModelRegistry(SETTINGS.models_dir)
    for t in TASKS:
        v = reg.latest_version(t)
        st.markdown(f"`{t}`  \n<small>{v or 'not trained'}</small>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption(f"EduPulse v{__version__}. FastAPI, Streamlit, scikit-learn.")

PAGES[choice]()
