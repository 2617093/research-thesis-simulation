import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

st.set_page_config(page_title="MOF CO₂ Screening Simulator", page_icon="🧪", layout="wide")

st.title("MOF CO₂ Computational Screening Simulator")
st.caption("16-descriptor framework | Structural Properties (SP) · Chemical Properties (CP) · Adsorption Properties (AP)")

st.info(
    "Research prototype trained from the values supplied in your PDFs. "
    "The matched training table contains 12 MOFs. Predictions are preliminary model estimates, "
    "not new GCMC results or experimentally validated values."
)

BASE_DATA = {
    "mof_name": ["hMOF-5 (IRMOF-1)", "MOF-199 (Cu-BTC)", "Mg-MOF-74", "CPM-5", "MOF-508",
                 "NU-100", "NU-1000", "ZIF-11", "SDU-1", "UMCM-1", "ZIF-71", "IRMOF-3"],
    "topology": ["pcu","tbo","ETY","nbo/flu","fsc","rht","csq","RHO","unknown","umc","RHO","pcu"],
    "pore_volume_cm3_g": [1.31,0.75,0.57,0.52,0.29,2.82,1.39,0.44,0.32,1.41,0.35,1.22],
    "accessible_surface_area_m2_g": [3500,1800,1500,2187,743,6140,2320,1680,960,4160,1050,2450],
    "void_fraction": [0.80,0.68,0.51,0.56,0.35,0.86,0.72,0.48,0.41,0.77,0.43,0.76],
    "pld_A": [11.0,6.9,11.0,5.2,4.0,13.4,12.0,3.0,4.2,12.0,3.4,11.0],
    "lcd_A": [15.1,13.2,11.0,13.0,4.0,27.4,31.0,14.6,5.5,14.0,14.8,15.1],
    "crystal_density_g_cm3": [0.59,0.88,0.91,1.05,1.25,0.38,0.49,0.83,1.31,0.61,0.85,0.62],
    "metal_node": ["Zn","Cu","Mg","In","Zn","Cu","Zr","Zn","Al","Zn","Zn","Zn"],
    "linker_functional_group": ["BDC","BTC","DOBDC","BTC","BDC; BPTC","Hexatopic","TBAPy",
                                "PhIm","Hydroquinone","BDC; BTB","dcIm","BDC-NH2"],
    "co2_uptake_mmol_g": [2.25,2.99,8.79,2.00,1.30,1.70,1.70,1.05,1.45,2.11,0.85,3.14],
    "henry_coefficient_mmol_kg_bar": [0.64,0.79,1.42,0.59,0.42,0.51,0.53,0.38,0.46,0.68,0.31,0.87],
    "co2_n2_selectivity": [8.9,11.8,18.7,8.4,6.3,7.5,7.8,5.8,6.9,9.7,4.9,12.9],
}
built_in = pd.DataFrame(BASE_DATA)

@st.cache_resource
def train_model(data):
    numeric_features = [
        "pore_volume_cm3_g", "accessible_surface_area_m2_g", "void_fraction",
        "pld_A", "lcd_A", "crystal_density_g_cm3"
    ]
    categorical_features = ["metal_node", "linker_functional_group"]
    targets = ["co2_uptake_mmol_g", "henry_coefficient_mmol_kg_bar", "co2_n2_selectivity"]

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("categorical", categorical_pipe, categorical_features)
    ])
    model = Pipeline([
        ("prep", prep),
        ("regressor", MultiOutputRegressor(
            RandomForestRegressor(n_estimators=350, min_samples_leaf=1, random_state=42)
        ))
    ])
    X = data[numeric_features + categorical_features]
    y = data[targets]
    model.fit(X, y)
    return model, numeric_features, categorical_features, targets

st.sidebar.header("Training data")
uploaded = st.sidebar.file_uploader("Optional: upload a larger training CSV", type=["csv"])
training = built_in.copy()
if uploaded is not None:
    try:
        extra = pd.read_csv(uploaded)
        required = list(built_in.columns)
        missing = [c for c in required if c not in extra.columns]
        if missing:
            st.sidebar.error("CSV missing columns: " + ", ".join(missing))
        else:
            training = pd.concat([training, extra[required]], ignore_index=True)
            st.sidebar.success(f"Added {len(extra)} rows to the supplied 12-row dataset.")
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")

model, numeric_features, categorical_features, targets = train_model(training)
st.sidebar.metric("Training records", len(training))
st.sidebar.caption("Built-in records: 12 matched MOFs. Upload additional records to improve coverage.")

st.subheader("1. Enter a new MOF")
mof_name = st.text_input("MOF name or candidate ID", value="New MOF candidate")

sp_tab, cp_tab, ap_tab = st.tabs(["SP · Structural Properties", "CP · Chemical Properties", "AP · Adsorption Properties"])

with sp_tab:
    st.markdown("Enter the six structural descriptors. Units must match the labels.")
    c1, c2, c3 = st.columns(3)
    with c1:
        asa = st.number_input("Accessible surface area (m²/g)", min_value=0.0, value=2000.0, step=50.0)
        pv = st.number_input("Pore volume (cm³/g)", min_value=0.0, value=0.80, step=0.05, format="%.3f")
    with c2:
        vf = st.number_input("Void fraction (0–1)", min_value=0.0, max_value=1.0, value=0.60, step=0.01)
        pld = st.number_input("Pore-limiting diameter, PLD (Å)", min_value=0.0, value=7.0, step=0.1)
    with c3:
        lcd = st.number_input("Largest cavity diameter, LCD (Å)", min_value=0.0, value=12.0, step=0.1)
        density = st.number_input("Crystal density (g/cm³)", min_value=0.001, value=0.90, step=0.01)

with cp_tab:
    st.markdown("Enter the four chemical descriptors. Metal and linker are used by the current trained model.")
    c1, c2 = st.columns(2)
    with c1:
        metal = st.selectbox("Metal identity / node type", ["Zn","Cu","Mg","Co","Ni","Fe","Al","Zr","Cr","In","Other"])
        linker = st.text_input("Organic linker / functional-group composition", value="BDC-NH2",
                               help="Use a consistent name, e.g. BDC, BTC, BDC-NH2, DOBDC.")
    with c2:
        oms_choice = st.selectbox("Open-metal-site presence (OMS)", ["Unknown","Present","Absent"])
        charge_value = st.text_input("Partial atomic charges", value="",
            help="Optional: enter a documented summary, e.g. mean framework atomic charge. The model does not use this field because no charge labels were supplied.")
    st.caption("OMS and partial-charge values are displayed for the full descriptor record, but are not model predictors because the supplied training PDFs do not provide these values consistently.")

with ap_tab:
    st.markdown("Enter known operating/adsorption context. These fields are recorded but are not used as predictors in this trained model.")
    c1, c2, c3 = st.columns(3)
    with c1:
        qst = st.number_input("Heat of adsorption, Qst (kJ/mol) — optional", min_value=0.0, value=0.0, step=1.0)
        wc = st.number_input("Working capacity (mmol/g) — optional", min_value=0.0, value=0.0, step=0.1)
    with c2:
        pressure = st.number_input("Operating pressure (bar)", min_value=0.0, value=1.0, step=0.1)
        temperature = st.number_input("Operating temperature (K)", min_value=0.0, value=298.0, step=1.0)
    with c3:
        gas_co2 = st.number_input("Gas-phase CO₂ fraction", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
        gas_n2 = st.number_input("Gas-phase N₂ fraction", min_value=0.0, max_value=1.0, value=0.85, step=0.01)
    st.caption("The model predicts uptake, Henry coefficient and selectivity because those are the three targets supplied in the uploaded screening table. Qst and working capacity cannot be trained from the current PDFs.")

st.divider()
st.subheader("2. Descriptor calculations")
# SP score is an illustrative normalised composite, not a physical property.
med = training[["accessible_surface_area_m2_g","pore_volume_cm3_g","void_fraction","pld_A","lcd_A","crystal_density_g_cm3"]].median()
sp_terms = [
    asa / med["accessible_surface_area_m2_g"],
    pv / med["pore_volume_cm3_g"],
    vf / med["void_fraction"],
    pld / med["pld_A"],
    lcd / med["lcd_A"],
    med["crystal_density_g_cm3"] / density
]
sp_score = float(np.mean(sp_terms))
cp_known = sum([bool(metal), bool(linker.strip()), oms_choice != "Unknown", bool(charge_value.strip())])
cp_completeness = 100 * cp_known / 4
m1, m2, m3 = st.columns(3)
m1.metric("SP normalised composite score", f"{sp_score:.3f}")
m2.metric("CP descriptor completeness", f"{cp_completeness:.0f}%")
m3.metric("AP conditions entered", "Yes" if pressure > 0 and temperature > 0 else "Incomplete")
st.caption("SP score is a transparent normalised screening index using training-set medians; it is not an experimentally defined MOF property. CP completeness reports how many of the four chemical fields have been supplied, not a chemical-performance score.")

st.subheader("3. Machine-learning prediction")
if st.button("Run CO₂ screening prediction", type="primary", use_container_width=True):
    if asa <= 0 or pv <= 0 or density <= 0 or pld <= 0 or lcd <= 0:
        st.error("Please enter positive values for surface area, pore volume, PLD, LCD and density.")
    elif not linker.strip():
        st.error("Please enter the organic linker / functional-group composition.")
    else:
        metal_model = metal if metal != "Other" else "Other"
        candidate = pd.DataFrame([{
            "pore_volume_cm3_g": pv,
            "accessible_surface_area_m2_g": asa,
            "void_fraction": vf,
            "pld_A": pld,
            "lcd_A": lcd,
            "crystal_density_g_cm3": density,
            "metal_node": metal_model,
            "linker_functional_group": linker.strip()
        }])
        pred = model.predict(candidate)[0]
        uptake, henry, selectivity = [float(x) for x in pred]
        # Prevent impossible negative outputs from a regression model.
        uptake, henry, selectivity = max(0.0, uptake), max(0.0, henry), max(0.0, selectivity)

        st.success(f"Screening prediction completed for {mof_name}.")
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted CO₂ uptake", f"{uptake:.2f} mmol/g")
        r2.metric("Predicted Henry coefficient", f"{henry:.3f} mmol/kg/bar")
        r3.metric("Predicted CO₂/N₂ selectivity", f"{selectivity:.2f}")

        # Transparent screening band based on training target distribution; not a universal threshold.
        q75 = training["co2_uptake_mmol_g"].quantile(0.75)
        q25 = training["co2_uptake_mmol_g"].quantile(0.25)
        if uptake >= q75:
            verdict = "Higher predicted uptake within this small reference dataset"
        elif uptake >= q25:
            verdict = "Intermediate predicted uptake within this small reference dataset"
        else:
            verdict = "Lower predicted uptake within this small reference dataset"
        st.markdown(f"**Preliminary screening interpretation:** {verdict}.")
        st.caption(f"Comparison bands are based on the supplied training uptake distribution (25th percentile {q25:.2f}; 75th percentile {q75:.2f} mmol/g), not universal pass/fail criteria.")

        result = pd.DataFrame([{
            "MOF_candidate": mof_name,
            "SP_composite_score": sp_score,
            "CP_completeness_percent": cp_completeness,
            "CO2_uptake_pred_mmol_g": uptake,
            "Henry_coefficient_pred_mmol_kg_bar": henry,
            "CO2_N2_selectivity_pred": selectivity,
            "Qst_user_entered_kJ_mol": qst if qst > 0 else np.nan,
            "working_capacity_user_entered_mmol_g": wc if wc > 0 else np.nan,
            "pressure_bar": pressure,
            "temperature_K": temperature,
            "gas_CO2_fraction": gas_co2,
            "gas_N2_fraction": gas_n2,
            "model_training_rows": len(training),
            "model": "Random Forest multi-output regression"
        }])
        st.dataframe(result, use_container_width=True)
        st.download_button("Download prediction as CSV", result.to_csv(index=False).encode("utf-8"),
                           file_name="mof_screening_prediction.csv", mime="text/csv")

st.divider()
st.subheader("4. Supplied training dataset")
st.write("The following 12 MOFs are the records matched across the uploaded structural database and adsorption table.")
st.dataframe(training, use_container_width=True, hide_index=True)
st.download_button("Download training dataset CSV", training.to_csv(index=False).encode("utf-8"),
                   file_name="mof_training_data.csv", mime="text/csv")

with st.expander("Model scope, assumptions and limitations"):
    st.markdown("""
    **Training source and scope**
    - Structural inputs (pore volume, accessible surface area, void fraction, PLD, LCD, density, metal and linker) are taken from the supplied MOF property database.
    - Output labels (CO₂ uptake, Henry coefficient and CO₂/N₂ selectivity) are taken from the supplied 15-candidate screening table for the 12 MOFs that match the structural database.
    - The uploaded report describes these adsorption metrics as machine-learning predicted screening metrics. They should therefore be treated as supplied reference labels, not independently verified experimental measurements.

    **Why not all 16 descriptors are model predictors**
    - OMS and partial atomic charges are not consistently supplied in the training PDFs.
    - Qst and working capacity have no target values in the supplied training table.
    - The uploaded table has no consistent pressure/temperature metadata for each row.
    - Henry coefficient, uptake and selectivity are model outputs here, so they are not fed back as inputs when predicting themselves (avoids target leakage).

    **Scientific caution**
    - Twelve training records are far too few for a reliable general-purpose MOF model. Predictions for unseen metals/linkers may be extrapolations.
    - The model is a working prototype for software development and exploratory screening, not a substitute for GCMC/IAST calculations or experimental validation.
    - Add a larger, consistently standardised dataset before using predictions for research conclusions.
    """)

st.caption("Project: Computational Screening of Metal–Organic Frameworks for the Adsorption of Carbon Dioxide.")

