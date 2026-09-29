import streamlit as st
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


                                                           
             
                                                           
st.set_page_config(
    page_title="CreditWise",
    page_icon="💳",
    layout="wide"
)

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #173b67;
    }

    .sub-title {
        font-size: 18px;
        color: #64748b;
        margin-bottom: 25px;
    }

    .card {
        padding: 25px;
        border-radius: 15px;
        background-color: #F1DFD3;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.0);
        margin-bottom: 20px;
        color:black;
    }

    .approved {
        padding: 25px;
        border-radius: 15px;
        color:black;
        background-color: #dcfce7;
        border-left: 6px solid #16a34a;
    }

    .rejected {
        padding: 25px;
        border-radius: 15px;
        color:black;
        background-color: #fee2e2;
        border-left: 6px solid #dc2626;
    }

    .result {
        font-size: 30px;
        font-weight: 800;
    }

    </style>
    """,
    unsafe_allow_html=True
)


                                                           
           
                                                           

@st.cache_data
def load_data():

    df = pd.read_csv("loan_approval_data.csv")

    return df


df_original = load_data()


                                                           
              
                                                           

@st.cache_resource
def train_models(data):

    df = data.copy()

                                                           
                                         
                                                           

    df = df.dropna(subset=["Loan_Approved"]).copy()

                                                           
                   
                                                           

    numerical_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    
    categorical_cols = df.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

                                                      
    if "Loan_Approved" in numerical_cols:
        numerical_cols.remove("Loan_Approved")

    if "Loan_Approved" in categorical_cols:
        categorical_cols.remove("Loan_Approved")

                                                           
                              
                            
                                                           

    num_imputer = SimpleImputer(
        strategy="mean"
    )

    df[numerical_cols] = num_imputer.fit_transform(
        df[numerical_cols]
    )

                                                           
                                
                                     
                                                           

    cat_imputer = SimpleImputer(
        strategy="most_frequent"
    )

    df[categorical_cols] = cat_imputer.fit_transform(
        df[categorical_cols]
    )

                                                           
                              
                                                           

    education_encoder = LabelEncoder()

    df["Education_Level"] = education_encoder.fit_transform(
        df["Education_Level"]
    )

                                                           
                     
     
            
             
                                                           

    df["Loan_Approved"] = (
        df["Loan_Approved"]
        .astype(str)
        .str.strip()
        .map({
            "No": 0,
            "Yes": 1
        })
    )

                  
    if df["Loan_Approved"].isnull().any():

        raise ValueError(
            "Loan_Approved contains values other than Yes/No."
        )

    df["Loan_Approved"] = df["Loan_Approved"].astype(int)

                                                           
                      
                              
                                                           

    ohe_columns = [
        "Employment_Status",
        "Marital_Status",
        "Loan_Purpose",
        "Property_Area",
        "Gender",
        "Employer_Category"
    ]

    ohe = OneHotEncoder(
        drop=None,
        handle_unknown="ignore",
        sparse_output=False
    )

    encoded = ohe.fit_transform(
        df[ohe_columns]
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(
            ohe_columns
        ),
        index=df.index
    )

    df = pd.concat(
        [
            encoded_df,
            df.drop(columns=ohe_columns)
        ],
        axis=1
    )

                                                           
                         
                      
                                                           

    df["Credit_Score_sq"] = (
        df["Credit_Score"] ** 2
    )

    df["DTI_Ratio_sq"] = (
        df["DTI_Ratio"] ** 2
    )

                                                           
                       
                                                           

    X = df.drop(
        columns=[
            "Loan_Approved",
            "Credit_Score",
            "DTI_Ratio"
        ]
    )

    y = df["Loan_Approved"]

                                                           
                      
                                                           

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

                                                           
                    
                      
                                                           

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

                                                           
                         
                                                           

    logistic_model = LogisticRegression()

    logistic_model.fit(
        X_train_scaled,
        y_train
    )

                                                           
         
                                 
                                                           

    knn_model = KNeighborsClassifier(
        n_neighbors=5
    )

    knn_model.fit(
        X_train_scaled,
        y_train
    )

                                                           
                 
                                                           

    nb_model = GaussianNB()

    nb_model.fit(
        X_train_scaled,
        y_train
    )

                                                           
            
                                                           

    models = {
        "Logistic Regression": logistic_model,
        "KNN": knn_model,
        "Naive Bayes": nb_model
    }

                                                           
                      
                                                           

    metrics = {}

    for model_name, model in models.items():

        prediction = model.predict(
            X_test_scaled
        )

        metrics[model_name] = {

            "Accuracy": accuracy_score(
                y_test,
                prediction
            ),

            "Precision": precision_score(
                y_test,
                prediction,
                average="binary",
                zero_division=0
            ),

            "Recall": recall_score(
                y_test,
                prediction,
                average="binary",
                zero_division=0
            ),

            "F1 Score": f1_score(
                y_test,
                prediction,
                average="binary",
                zero_division=0
            ),

            "Confusion Matrix": confusion_matrix(
                y_test,
                prediction
            )
        }

    return (
        models,
        scaler,
        ohe,
        education_encoder,
        num_imputer,
        cat_imputer,
        metrics,
        X.columns.tolist()
    )


                                                           
       
                                                           

(
    models,
    scaler,
    ohe,
    education_encoder,
    num_imputer,
    cat_imputer,
    metrics,
    feature_columns
) = train_models(df_original)


                                                           
         
                                                           

st.sidebar.title("💳 CreditWise")

st.sidebar.write(
    "Loan Approval Prediction System"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔮 Loan Prediction",
        "📊 Model Performance",
        "📈 Dataset"
    ]
)


                                                           
      
                                                           

if page == "🏠 Home":

    st.markdown(
        '<div class="main-title">💳 CreditWise</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">'
        'Machine Learning Based Loan Approval Prediction System'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">

        <h3>🎯 About CreditWise</h3>

        CreditWise predicts whether a loan application
        is likely to be approved based on applicant,
        financial and loan-related information.

        </div>
        """,
        unsafe_allow_html=True
    )

                                                           
                
                                                           

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Applications",
            len(df_original)
        )

    with col2:
        st.metric(
            "Features",
            len(df_original.columns) - 1
        )

    with col3:
        st.metric(
            "Models",
            3
        )

    with col4:

        approved = (
            df_original["Loan_Approved"]
            .astype(str)
            .eq("Yes")
            .sum()
        )

        st.metric(
            "Approved",
            int(approved)
        )

    st.markdown("---")

                                                           
              
                                                           

    st.subheader("🔄 Machine Learning Workflow")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.info("📂 Dataset")

    with c2:
        st.info("🧹 Cleaning")

    with c3:
        st.info("⚙️ Encoding")

    with c4:
        st.info("🤖 Training")

    with c5:
        st.info("🎯 Prediction")

    st.markdown("---")

    st.subheader("🤖 Models Used")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="card">

            ### Logistic Regression

            Classification model used for
            loan approval prediction.

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="card">

            ### KNN

            K-Nearest Neighbors classifier
            with 5 neighbors.

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="card">

            ### Naive Bayes

            Gaussian Naive Bayes
            classification model.

            </div>
            """,
            unsafe_allow_html=True
        )


                                                           
                 
                                                           

elif page == "🔮 Loan Prediction":

    st.title("🔮 Loan Approval Prediction")

    st.write(
        "Enter the applicant details below."
    )

    st.markdown("---")

                                                           
               
                                                           

    st.subheader("👤 Applicant Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        applicant_id = st.number_input(
            "Applicant ID",
            min_value=1,
            value=1001,
            step=1
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

        gender = st.selectbox(
            "Gender",
            sorted(
                df_original["Gender"]
                .dropna()
                .unique()
                .tolist()
            )
        )

    with col2:

        marital_status = st.selectbox(
            "Marital Status",
            sorted(
                df_original["Marital_Status"]
                .dropna()
                .unique()
                .tolist()
            )
        )

        dependents = st.number_input(
            "Dependents",
            min_value=0,
            max_value=10,
            value=0
        )

        education = st.selectbox(
            "Education Level",
            sorted(
                df_original["Education_Level"]
                .dropna()
                .unique()
                .tolist()
            )
        )

    with col3:

        employment = st.selectbox(
            "Employment Status",
            sorted(
                df_original["Employment_Status"]
                .dropna()
                .unique()
                .tolist()
            )
        )

        employer_category = st.selectbox(
            "Employer Category",
            sorted(
                df_original["Employer_Category"]
                .dropna()
                .unique()
                .tolist()
            )
        )

                                                           
               
                                                           

    st.markdown("---")

    st.subheader("💰 Financial Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0.0,
            value=10000.0,
            step=500.0
        )

        coapplicant_income = st.number_input(
            "Coapplicant Income",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

        savings = st.number_input(
            "Savings",
            min_value=0.0,
            value=10000.0,
            step=500.0
        )

    with col2:

        existing_loans = st.number_input(
            "Existing Loans",
            min_value=0,
            max_value=20,
            value=1
        )

        credit_score = st.number_input(
            "Credit Score",
            min_value=300.0,
            max_value=900.0,
            value=700.0,
            step=1.0
        )

        dti_ratio = st.number_input(
            "DTI Ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.30,
            step=0.01
        )

    with col3:

        collateral_value = st.number_input(
            "Collateral Value",
            min_value=0.0,
            value=25000.0,
            step=1000.0
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0.0,
            value=20000.0,
            step=1000.0
        )

        loan_term = st.selectbox(
            "Loan Term",
            [12, 24, 36, 48, 60, 72, 84],
            index=3
        )

                                                           
          
                                                           

    st.markdown("---")

    st.subheader("🏦 Loan Information")

    col1, col2 = st.columns(2)

    with col1:

        loan_purpose = st.selectbox(
            "Loan Purpose",
            sorted(
                df_original["Loan_Purpose"]
                .dropna()
                .unique()
                .tolist()
            )
        )

    with col2:

        property_area = st.selectbox(
            "Property Area",
            sorted(
                df_original["Property_Area"]
                .dropna()
                .unique()
                .tolist()
            )
        )

                                                           
             
                                                           

    st.markdown("---")

    predict = st.button(
        "🚀 Predict Loan Approval",
        type="primary",
        use_container_width=True
    )

    if predict:

                                                           
                         
                                                           

        input_data = pd.DataFrame(
            {
                "Applicant_ID": [applicant_id],
                "Applicant_Income": [applicant_income],
                "Coapplicant_Income": [coapplicant_income],
                "Employment_Status": [employment],
                "Age": [age],
                "Marital_Status": [marital_status],
                "Dependents": [dependents],
                "Credit_Score": [credit_score],
                "Existing_Loans": [existing_loans],
                "DTI_Ratio": [dti_ratio],
                "Savings": [savings],
                "Collateral_Value": [collateral_value],
                "Loan_Amount": [loan_amount],
                "Loan_Term": [loan_term],
                "Loan_Purpose": [loan_purpose],
                "Property_Area": [property_area],
                "Education_Level": [education],
                "Gender": [gender],
                "Employer_Category": [employer_category]
            }
        )

                                                           
                              
                                                           

        input_data[
            num_imputer.feature_names_in_
        ] = num_imputer.transform(
            input_data[
                num_imputer.feature_names_in_
            ]
        )

                                                           
                                
                                                           

        input_data[
            cat_imputer.feature_names_in_
        ] = cat_imputer.transform(
            input_data[
                cat_imputer.feature_names_in_
            ]
        )

                                                           
                            
                                                           

        input_data["Education_Level"] = (
            education_encoder.transform(
                input_data["Education_Level"]
            )
        )

                                                           
                          
                                                           

        ohe_columns = [
            "Employment_Status",
            "Marital_Status",
            "Loan_Purpose",
            "Property_Area",
            "Gender",
            "Employer_Category"
        ]

        encoded = ohe.transform(
            input_data[ohe_columns]
        )

        encoded_df = pd.DataFrame(
            encoded,
            columns=ohe.get_feature_names_out(
                ohe_columns
            )
        )

        input_data = pd.concat(
            [
                encoded_df,
                input_data.drop(
                    columns=ohe_columns
                ).reset_index(drop=True)
            ],
            axis=1
        )

                                                           
                             
                                                           

        input_data["Credit_Score_sq"] = (
            input_data["Credit_Score"] ** 2
        )

        input_data["DTI_Ratio_sq"] = (
            input_data["DTI_Ratio"] ** 2
        )

                                  
        input_data = input_data.drop(
            columns=[
                "Credit_Score",
                "DTI_Ratio"
            ]
        )

                                                           
                               
                                                           

        input_data = input_data.reindex(
            columns=feature_columns,
            fill_value=0
        )

                                                           
                 
                                                           

        input_scaled = scaler.transform(
            input_data
        )

                                                           
                                        
                                                           

        model = models[
            "Logistic Regression"
        ]

        prediction = model.predict(
            input_scaled
        )[0]

        probabilities = model.predict_proba(
            input_scaled
        )[0]

        approval_probability = (
            probabilities[1] * 100
        )

        rejection_probability = (
            probabilities[0] * 100
        )

                                                           
                
                                                           

        st.markdown("---")

        st.subheader("📋 Prediction Result")

        if prediction == 1:

            st.markdown(
                f"""
                <div class="approved">

                <div class="result">
                ✅ LOAN APPROVED
                </div>

                <br>

                Approval Probability:
                <strong>
                {approval_probability:.2f}%
                </strong>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="rejected">

                <div class="result">
                ❌ LOAN NOT APPROVED
                </div>

                <br>

                Approval Probability:
                <strong>
                {approval_probability:.2f}%
                </strong>

                </div>
                """,
                unsafe_allow_html=True
            )
        st.markdown("---")
        st.subheader("Possible area to Imporve ")
        approve_data=df_original[
            df_original["Loan_Approved"].astype(str).str.strip().eq("Yes")
        ].copy()
        suggestions=[]
        if not approve_data.empty:
            credit_score_median=approve_data["Credit_Score"].median()
            if(credit_score<credit_score_median):
                suggestions.append(
                    f"Your credit_score {credit_score} is less than credit_score_median {credit_score_median}.\n\n"
                    f"To Get an Loan you much increase your credit_Score "
                )
        applicant_income_median=approve_data["Applicant_Income"].median()
        if(applicant_income<applicant_income_median):
            suggestions.append(
                f"Your Income {applicant_income} is less than Application_income_median {applicant_income_median}\n\n"
                f"To Get the Loan you much increase your Income"
            )
        dti_ratio_median=approve_data["DTI_Ratio"].median()
        if(dti_ratio>dti_ratio_median):
            suggestions.append(
                f"Your Dead-to-Income ratio {dti_ratio} is greater than Dead-to-Income ratio {dti_ratio_median}\n\n"
                f"To Get the Loan you much decrease you Dead-to-Income ratio"
            )
        if suggestions:
            for suggestion in suggestions:
                st.info(suggestion)
        else:
            st.success(
                "No specific improvement area was identified "
                "from these three factors."
            )
        

                                                           
                     
                                                           

        st.markdown("---")

        st.subheader("📊 Prediction Probability")

        probability_df = pd.DataFrame(
            {
                "Status": [
                    "Not Approved",
                    "Approved"
                ],
                "Probability": [
                    rejection_probability,
                    approval_probability
                ]
            }
        )

        st.bar_chart(
            probability_df.set_index(
                "Status"
            )
        )

                                                           
                 
                                                           

        st.subheader("👤 Applicant Summary")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Age",
                age
            )

        with c2:
            st.metric(
                "Credit Score",
                f"{credit_score:.0f}"
            )

        with c3:
            st.metric(
                "Loan Amount",
                f"₹{loan_amount:,.0f}"
            )

        with c4:
            st.metric(
                "DTI Ratio",
                f"{dti_ratio:.2f}"
            )


                                                           
                   
                                                           

elif page == "📊 Model Performance":

    st.title("📊 Model Performance")

    st.write(
        "Evaluation results on the test dataset."
    )

                                                           
                       
                                                           

    performance = []

    for model_name, result in metrics.items():

        performance.append(
            {
                "Model": model_name,
                "Accuracy": result["Accuracy"],
                "Precision": result["Precision"],
                "Recall": result["Recall"],
                "F1 Score": result["F1 Score"]
            }
        )

    performance_df = pd.DataFrame(
        performance
    )

                                                           
           
                                                           

    display_df = performance_df.copy()

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]:

        display_df[column] = (
            display_df[column] * 100
        ).round(2).astype(str) + "%"

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

                                                           
           
                                                           

    st.subheader("📈 Model Comparison")

    st.bar_chart(
        performance_df.set_index(
            "Model"
        )
    )

                                                           
                      
                                                           

    st.markdown("---")

    st.subheader("🔢 Confusion Matrix")

    selected_model = st.selectbox(
        "Select Model",
        list(metrics.keys())
    )

    matrix = metrics[
        selected_model
    ]["Confusion Matrix"]

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            "Actual Not Approved",
            "Actual Approved"
        ],
        columns=[
            "Predicted Not Approved",
            "Predicted Approved"
        ]
    )

    st.dataframe(
        matrix_df,
        use_container_width=True
    )


                                                           
         
                                                           

elif page == "📈 Dataset":

    st.title("📈 Dataset Overview")

                                                           
                
                                                           

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Rows",
            df_original.shape[0]
        )

    with c2:
        st.metric(
            "Columns",
            df_original.shape[1]
        )

    with c3:
        st.metric(
            "Missing Values",
            int(
                df_original.isnull()
                .sum()
                .sum()
            )
        )

    with c4:

        approved = (
            df_original["Loan_Approved"]
            .astype(str)
            .eq("Yes")
            .sum()
        )

        st.metric(
            "Approved",
            int(approved)
        )

    st.markdown("---")

                                                           
             
                                                           

    st.subheader("📋 Dataset")

    st.dataframe(
        df_original,
        use_container_width=True,
        height=450
    )

                                                           
                                
                                                           

    st.subheader(
        "🎯 Loan Approval Distribution"
    )

    approval_counts = (
        df_original["Loan_Approved"]
        .value_counts()
    )

    st.bar_chart(
        approval_counts
    )

    st.subheader("🧹 Missing Values")

    missing = (
        df_original.isnull()
        .sum()
        .reset_index()
    )

    missing.columns = [
        "Column",
        "Missing Values"
    ]

    missing = missing[
        missing["Missing Values"] > 0
    ]

    if not missing.empty:

        st.dataframe(
            missing,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            "No missing values found."
        )

st.markdown("---")

st.caption(
    "CreditWise | Python + Pandas + Scikit-learn + Streamlit"
)