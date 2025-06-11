
import streamlit as st
import pandas as pd
import numpy as np
import pickle


# Load your trained CatBoost model
@st.cache_resource
def load_model():
    with open("app_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("Daily Revenue Share Predictor")

st.markdown(
    "Enter the details of the sailings to predict the daily share of revenue as a percentage."
)


# Excel upload section (one row per sailing, with comma-separated PORT_CODES)
st.subheader("Upload Sailings Excel File (One Row Per Sailing, Comma-Separated Port Codes)")

# Provide downloadable template for user reference
with open("RFA_App_Template.xlsx", "rb") as template_file:
    st.download_button(
        label="Download Excel Template",
        data=template_file.read(),
        file_name="RFA_App_Template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

excel_file = st.file_uploader("Upload an Excel file with all required columns", type=["xlsx"])

if excel_file is not None:
    sailing_df = pd.read_excel(excel_file)
    st.write("Preview of uploaded data:")
    st.dataframe(sailing_df.head(), use_container_width=True)

    # Expand each sailing into per-day rows using comma-separated PORT_CODES
    all_rows = []
    for idx, row in sailing_df.iterrows():
        dep_date_str = str(pd.to_datetime(row["DEPARTURE_DATE"]).date())
        sailing_id = f"{row['CASINO_CODE']}_{dep_date_str}"
        sailing_length = int(row["SAILING_LENGTH"])
        port_codes = [code.strip().upper() for code in str(row["PORT_CODES"]).split(",")]
        if len(port_codes) < sailing_length:
            st.error(f"Row {idx+1}: Number of port codes ({len(port_codes)}) is less than sailing length ({sailing_length}). Please fix the input file.")
            st.stop()
        elif len(port_codes) > sailing_length:
            st.error(f"Row {idx+1}: Number of port codes ({len(port_codes)}) is greater than sailing length ({sailing_length}). Please fix the input file.")
            st.stop()
        for day in range(1, sailing_length + 1):
            port_code = port_codes[day - 1]
            day_row = {
                "SAILING_ID": sailing_id,
                "CASINO_CODE": row["CASINO_CODE"],
                "DAY_NUMBER": day,
                "META_PRODUCT_CODE": row["META_PRODUCT_CODE"],
                "CURR_NUMBER_SAIL_NIGHTS": sailing_length,
                "PORT_CODE": port_code,
                "PERFECT_DAY_FLAG": port_code == "PCC",
                "DEPARTURE_MONTH": int(pd.to_datetime(row["DEPARTURE_DATE"]).month),
                "DEPARTURE_DAY_OF_WEEK": int(pd.to_datetime(row["DEPARTURE_DATE"]).weekday()),
                "DEPARTURE_YEAR": int(pd.to_datetime(row["DEPARTURE_DATE"]).year),
                "DEPARTURE_SEASON": int(pd.to_datetime(row["DEPARTURE_DATE"]).month) % 12 // 3,
                "BERTH_DAY_OF_WEEK": (pd.to_datetime(row["DEPARTURE_DATE"]).weekday() + day - 1) % 7,
                "IS_SEA_DAY": port_code == "CRU"
            }
            all_rows.append(day_row)
    input_df = pd.DataFrame(all_rows)

    # Ensure IS_SEA_DAY is boolean
    input_df["IS_SEA_DAY"] = input_df["IS_SEA_DAY"].astype(bool)

    feature_cols = [
        "CASINO_CODE",
        "DAY_NUMBER",
        "META_PRODUCT_CODE",
        "CURR_NUMBER_SAIL_NIGHTS",
        "PORT_CODE",
        "PERFECT_DAY_FLAG",
        "DEPARTURE_MONTH",
        "DEPARTURE_DAY_OF_WEEK",
        "DEPARTURE_YEAR",
        "DEPARTURE_SEASON",
        "BERTH_DAY_OF_WEEK",
        "IS_SEA_DAY"
    ]
    model_input_df = input_df[feature_cols]

    # Predict
    predictions = model.predict(model_input_df)
    input_df["Predicted Revenue Share"] = predictions

    # Normalize per sailing
    input_df["Predicted Revenue Share (%)"] = (
        input_df.groupby("SAILING_ID")["Predicted Revenue Share"].transform(lambda x: x / x.sum() * 100)
    ).round(2)

    # Display
    st.subheader("Batch Predicted Revenue Share Table")
    output_df = input_df[["SAILING_ID", "DAY_NUMBER", "Predicted Revenue Share (%)"]]
    st.dataframe(
        output_df,
        use_container_width=True,
        hide_index=True
    )

    # Export to Excel with user-specified filename
    excel_filename = st.text_input(
        "Enter Excel file name (without .xlsx)",
        value="predicted_revenue_share"
    )
    import io
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        output_df.to_excel(writer, index=False, sheet_name='Predictions')
    processed_data = output.getvalue()
    st.download_button(
        label="Download results as Excel",
        data=processed_data,
        file_name=f"{excel_filename}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

 