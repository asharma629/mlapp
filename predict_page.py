from operator import truediv
import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt



def load_data():
    data = pd.read_csv('model_all_data.csv')
    cols = [5,6]
    data.drop(data.columns[cols], axis=1,inplace=True)
    data['NZPBldgType'] = data['NZPBldgType'].astype('category')
    data = pd.get_dummies(data, columns=['NZPBldgType'])
    data['Characterized BldgSF'] = data['Characterized BldgSF'].str.replace(',', '').astype(float)
    data.dropna(inplace = True)
    return data

data = load_data()
mat_type_1 = data.loc[data['Material type'] == "#1"]
mat_type_2 = data.loc[data['Material type'] == "#2"]
mat_type_3 = data.loc[data['Material type'] == "#3"]
mat_type_4 = data.loc[data['Material type'] == "#4"]
mat_type_5 = data.loc[data['Material type'] == "#5"]
mat_type_6 = data.loc[data['Material type'] == "#6"]
mat_type_7 = data.loc[data['Material type'] == "#7"]
mat_type_aluminum = data.loc[data['Material type'] == "Aluminum"]
mat_type_corrcardboard = data.loc[data['Material type'] == "Corr. Cardboard"]
#mat_type_ewaste = data.loc[data['Material type'] == "E-waste"]
mat_type_food = data.loc[data['Material type'] == "Food"]
mat_type_glass = data.loc[data['Material type'] == "Glass"]
mat_type_mixedpaper = data.loc[data['Material type'] == "Mixed Paper"]
mat_type_newspaper = data.loc[data['Material type'] == "Newspaper"]
mat_type_nonrecyclmsw = data.loc[data['Material type'] == "Non-recyclable MSW"]
mat_type_paperboard = data.loc[data['Material type'] == "Paperboard"]
mat_type_soiledpaper = data.loc[data['Material type'] == "Soiled Paper"]
mat_type_steelferrous = data.loc[data['Material type'] == "Steel / Ferrous"]
#mat_type_textiles = data.loc[data['Material type'] == "Textiles"]
mat_type_whitepaper = data.loc[data['Material type'] == "White Paper"]
#mat_type_wood = data.loc[data['Material type'] == "Wood"]
mat_type_yardtrimmings = data.loc[data['Material type'] == "Yard Trimmings"]


# #1 plastics
X_train= mat_type_1.loc[:,['Characterized BldgSF', 'NZPBldgType_DFAC', 'NZPBldgType_UEPH']].astype('int')
y_train = mat_type_1['Total Pounds/Day']
X_train_coef = sm.tools.add_constant(X_train)
mat_type_1_model = sm.OLS(y_train, X_train_coef)
results1 = mat_type_1_model.fit()

# #2 plastics
X_train= mat_type_2.loc[:,['Characterized BldgSF', 'NZPBldgType_DFAC', 'NZPBldgType_UEPH']].astype('int')
y_train = mat_type_2['Total Pounds/Day']
X_train_coef = sm.tools.add_constant(X_train)
mat_type_2_model = sm.OLS(y_train, X_train_coef)
results2 = mat_type_2_model.fit()

# #3 plastics
X_train= mat_type_3.loc[:,['Characterized BldgSF','NZPBldgType_GIB']].astype('int')
y_train = mat_type_3['Total Pounds/Day']
X_train_coef = sm.tools.add_constant(X_train)

mat_type_3_model = sm.OLS(y_train, X_train_coef)
results3 = mat_type_3_model.fit()

# #4 plastics
X_train= mat_type_4.loc[:,['Characterized BldgSF', 'NZPBldgType_DFAC', 'NZPBldgType_PX']].astype('int')
y_train = mat_type_4['Total Pounds/Day']
X_train_coef = sm.tools.add_constant(X_train)
mat_type_4_model = sm.OLS(y_train, X_train_coef)
results4 = mat_type_4_model.fit()

# #5 plastics
X_train= mat_type_5.loc[:,['Characterized BldgSF', 'NZPBldgType_DFAC', 'NZPBldgType_PX']].astype('int')
y_train = mat_type_5['Total Pounds/Day']
X_train_coef = sm.tools.add_constant(X_train)
mat_type_5_model = sm.OLS(y_train, X_train_coef)
results5 = mat_type_5_model.fit()

# #6 plastics
X_train= mat_type_6.loc[:,['Characterized BldgSF', 'NZPBldgType_DFAC']].astype('int')
y_train = mat_type_6['Total Pounds/Day']
X_train_coef = sm.tools.add_constant(X_train)
mat_type_6_model = sm.OLS(y_train, X_train_coef)
results6 = mat_type_6_model.fit()

# #7 plastics
X_train= mat_type_7.loc[:,['Characterized BldgSF', 'NZPBldgType_DFAC']].astype('int')
y_train = mat_type_7['Total Pounds/Day']
X_train_coef = sm.tools.add_constant(X_train)
mat_type_7_model = sm.OLS(y_train, X_train_coef)
results7 = mat_type_7_model.fit()

# Aluminum
X_train= mat_type_aluminum.loc[:,['Characterized BldgSF', 'NZPBldgType_UEPH']].astype('int')
y_train = mat_type_aluminum['Total Pounds/Day']
X_train_coef = sm.tools.add_constant(X_train)
mat_type_aluminum_model = sm.OLS(y_train, X_train_coef)
results_aluminum = mat_type_aluminum_model.fit()

# Corr. Cardboard
X_train= mat_type_corrcardboard.loc[:,['Characterized BldgSF', 'NZPBldgType_DFAC', 'NZPBldgType_PX']].astype('int')
y_train = mat_type_corrcardboard['Total Pounds/Day']
X_train_coef = sm.tools.add_constant(X_train)
mat_type_corrcardboard_model = sm.OLS(y_train, X_train_coef)
results_corrcardboard = mat_type_corrcardboard_model.fit()

# Food
X_train= mat_type_food.loc[:,['Characterized BldgSF', 'NZPBldgType_DFAC', 'NZPBldgType_PX']].astype('int')
y_train = mat_type_food['Total Pounds/Day']
X_train_coef = sm.tools.add_constant(X_train)
mat_type_food_model = sm.OLS(y_train, X_train_coef)
results_food = mat_type_food_model.fit()

# Glass
X_train= mat_type_glass.loc[:,['Characterized BldgSF', 'NZPBldgType_Office-Large', 'NZPBldgType_UEPH']].astype('int')
y_train = mat_type_glass['Total Pounds/Day']
X_train_coef = sm.tools.add_constant(X_train)
mat_type_glass_model = sm.OLS(y_train, X_train_coef)
results_glass = mat_type_glass_model.fit()

# Mixed Paper
X_train= mat_type_mixedpaper.loc[:,['Characterized BldgSF', 'NZPBldgType_DFAC', 'NZPBldgType_Office-Large', 'NZPBldgType_PX', 'NZPBldgType_School-Primary']].astype('int')
y_train = mat_type_mixedpaper['Total Pounds/Day']
X_train_coef = sm.tools.add_constant(X_train)
mat_type_mixedpaper_model = sm.OLS(y_train, X_train_coef)
results_mixedpaper = mat_type_mixedpaper_model.fit()

# Newspaper
X_train= mat_type_newspaper.loc[:,['Characterized BldgSF', 'NZPBldgType_School-Primary', 'NZPBldgType_Training Barracks']].astype('int')
y_train = mat_type_newspaper['Total Pounds/Day']
X_train_coef = sm.tools.add_constant(X_train)
mat_type_newspaper_model = sm.OLS(y_train, X_train_coef)
results_newspaper = mat_type_newspaper_model.fit()

# Non-recycl MSW
X_train= mat_type_nonrecyclmsw.loc[:,['Characterized BldgSF', 'NZPBldgType_GIB', 'NZPBldgType_Office-Large', 'NZPBldgType_PX', 'NZPBldgType_School-Primary', 'NZPBldgType_Training Barracks']].astype('int')
y_train = mat_type_nonrecyclmsw['Total Pounds/Day']
X_train_coef = sm.tools.add_constant(X_train)
mat_type_nonrecyclmsw_model = sm.OLS(y_train, X_train_coef)
results_nonrecyclmsw = mat_type_nonrecyclmsw_model.fit()

# Paperboard
X_train= mat_type_paperboard.loc[:,['Characterized BldgSF', 'NZPBldgType_GIB', 'NZPBldgType_Office-Large', 'NZPBldgType_School-Primary']].astype('int')
y_train = mat_type_paperboard['Total Pounds/Day']
X_train_coef = sm.tools.add_constant(X_train)
mat_type_paperboard_model = sm.OLS(y_train, X_train_coef)
results_paperboard = mat_type_paperboard_model.fit()

# Soiled Paper
X_train= mat_type_soiledpaper.loc[:,['Characterized BldgSF', 'NZPBldgType_GIB', 'NZPBldgType_Office-Small', 'NZPBldgType_PFF', 'NZPBldgType_School-Primary', 'NZPBldgType_Training Barracks', 'NZPBldgType_UEPH']].astype('int')
y_train = mat_type_soiledpaper['Total Pounds/Day']
X_train_coef = sm.tools.add_constant(X_train)
mat_type_soiledpaper_model = sm.OLS(y_train, X_train_coef)
results_soiledpaper = mat_type_soiledpaper_model.fit()

# Steel / Ferrous
X_train= mat_type_steelferrous.loc[:,['Characterized BldgSF', 'NZPBldgType_DFAC']].astype('int')
y_train = mat_type_steelferrous['Total Pounds/Day']
X_train_coef = sm.tools.add_constant(X_train)
mat_type_steelferrous_model = sm.OLS(y_train, X_train_coef)
results_steelferrous = mat_type_steelferrous_model.fit()

# White Paper
X_train= mat_type_whitepaper.loc[:,['Characterized BldgSF', 'NZPBldgType_GIB', 'NZPBldgType_Office-Large', 'NZPBldgType_PX', 'NZPBldgType_School-Primary']].astype('int')
y_train = mat_type_whitepaper['Total Pounds/Day']
X_train_coef = sm.tools.add_constant(X_train)
mat_type_whitepaper_model = sm.OLS(y_train, X_train_coef)
results_whitepaper = mat_type_whitepaper_model.fit()

# Yard Trimmings 
X_train= mat_type_yardtrimmings.loc[:,['Characterized BldgSF', 'NZPBldgType_UEPH']].astype('int')
y_train = mat_type_yardtrimmings['Total Pounds/Day']
X_train_coef = sm.tools.add_constant(X_train)
mat_type_yardtrimmings_model = sm.OLS(y_train, X_train_coef)
results_yardtrimmings = mat_type_yardtrimmings_model.fit()

def show_predict_page():
    st.title("Solid Waste Forecasting")

    st.write("""### We will need the following information to make predictions: """)
    bldg_type = {"NZPBldgType_CDC", "NZPBldgType_DFAC", "NZPBldgType_GIB", "NZPBldgType_Office-Large", "NZPBldgType_Office-Medium", "NZPBldgType_Office-Small", "NZPBldgType_PFF", "NZPBldgType_PX", "NZPBldgType_School-Primary", "NZPBldgType_Training Barracks", "NZPBldgType_UEPH"}
    material = st.selectbox("Building Type", bldg_type)
    sq_footage = st.text_input("Square Footage", placeholder = "input total building square footage")
    submit = st.button("Forecast Values")

    if submit:
        if bldg_type == "NZPBldgType_DFAC":
            mat_1 = np.array([1, float(sq_footage), 1, 0])
            mat_2 = np.array([1, float(sq_footage), 1, 0])
            mat_4 = np.array([1, float(sq_footage), 1, 0])
            mat_5 = np.array([1, float(sq_footage), 1, 0])
            mat_6 = np.array([1, float(sq_footage), 1])
            mat_7 = np.array([1, float(sq_footage), 1])
            corr_cardboard = np.array([1, float(sq_footage), 1, 0])
            food = np.array([1, float(sq_footage), 1, 0])
            mixed_paper = np.array([1, float(sq_footage), 1, 0, 0, 0])
            steelferrous = np.array([1, float(sq_footage), 1])
        elif bldg_type == "NZPBldgType_GIB":
            mat_3 = np.array([1, float(sq_footage), 1])
            nonrecyclmsw = np.array([1, float(sq_footage), 1, 0, 0, 0, 0])
            paperboard = np.array([1, float(sq_footage), 1, 0, 0])
            soiledpaper = np.array([1, float(sq_footage), 1, 0, 0, 0, 0, 0])
            whitepaper = np.array([1, float(sq_footage), 1, 0, 0, 0])
        elif bldg_type == "NZPBldgType_Office-Large":
            glass = np.array([1, float(sq_footage), 1, 0])
            mixed_paper = np.array([1, float(sq_footage), 0, 1, 0, 0])
            nonrecyclmsw = np.array([1, float(sq_footage), 0, 1, 0, 0, 0])
            paperboard = np.array([1, float(sq_footage), 0, 1, 0])
            whitepaper = np.array([1, float(sq_footage), 0, 1, 0, 0])
        elif bldg_type == "NZPBldgType_Office-Small":
            soiledpaper = np.array([1, float(sq_footage), 0, 1, 0, 0, 0, 0])
        elif bldg_type == "NZPBldgType_PFF":
            soiledpaper = np.array([1, float(sq_footage), 0, 0, 1, 0, 0, 0])
        elif bldg_type == "NZPBldgType_UEPH":
            mat_1 = np.array([1, float(sq_footage), 0, 1])
            mat_2 = np.array([1, float(sq_footage), 0, 1])
            aluminum = np.array([1, float(sq_footage), 1])
            glass = np.array([1, float(sq_footage), 0, 1])
        elif bldg_type == "NZPBldgType_PX":
            mat_4 = np.array([1, float(sq_footage), 0, 1])
            mat_5 = np.array([1, float(sq_footage), 0, 1])
            corr_cardboard = np.array([1, float(sq_footage), 0, 1])
            food = np.array([1, float(sq_footage), 0, 1])
            mixed_paper = np.array([1, float(sq_footage), 0, 0, 1, 0])
            nonrecyclmsw = np.array([1, float(sq_footage), 0, 0, 1, 0, 0])
            whitepaper = np.array([1, float(sq_footage), 0, 0, 1, 0])
        elif bldg_type == "NZPBldgType_School-Primary":
            mixed_paper = np.array([1, float(sq_footage), 0, 0, 0, 1])
            newspaper = np.array([1, float(sq_footage), 1, 0])
            nonrecyclmsw = np.array([1, float(sq_footage), 0, 0, 0, 1, 0])
            paperboard = np.array([1, float(sq_footage), 0, 0, 1])
            soiledpaper = np.array([1, float(sq_footage), 0, 0, 0, 1, 0, 0])
            whitepaper = np.array([1, float(sq_footage), 0, 0, 0, 1])
        elif bldg_type == "NZPBldgType_Training Barracks":
            newspaper = np.array([1, float(sq_footage), 0, 1])
            nonrecyclmsw = np.array([1, float(sq_footage), 0, 0, 0, 0, 1])
            soiledpaper = np.array([1, float(sq_footage), 0, 0, 0, 0, 1, 0])
        elif bldg_type == "NZPBldgType_UEPH":
            soiledpaper = np.array([1, float(sq_footage), 0, 0, 0, 0, 0, 1])
            yardtrimmings = np.array([1, float(sq_footage), 1])
        else:
            mat_1 = np.array([1, float(sq_footage), 0, 0])
            mat_2 = np.array([1, float(sq_footage), 0, 0])
            mat_3 = np.array([1, float(sq_footage), 0])
            mat_4 = np.array([1, float(sq_footage), 0, 0])
            mat_5 = np.array([1, float(sq_footage), 0, 0])
            mat_6 = np.array([1, float(sq_footage), 0])
            mat_7 = np.array([1, float(sq_footage), 0])
            aluminum = np.array([1, float(sq_footage), 0])
            corr_cardboard = np.array([1, float(sq_footage), 0, 0])
            food = np.array([1, float(sq_footage), 0, 0])
            glass = np.array([1, float(sq_footage), 0, 0])
            mixed_paper = np.array([1, float(sq_footage), 0, 0, 0, 0])
            newspaper = np.array([1, float(sq_footage), 0, 0])
            nonrecyclmsw = np.array([1, float(sq_footage), 0, 0, 0, 0, 0])
            paperboard = np.array([1, float(sq_footage), 0, 0, 0])
            soiledpaper = np.array([1, float(sq_footage), 0, 0, 0, 0, 0, 0])
            steelferrous = np.array([1, float(sq_footage), 0])
            whitepaper = np.array([1, float(sq_footage), 0, 0, 0, 0])
            yardtrimmings = np.array([1, float(sq_footage), 0])


        mat_1_pred = results1.predict(mat_1)[0]

        mat_2_pred = results2.predict(mat_2)[0]

        mat_3_pred = results3.predict(mat_3)[0]

        mat_4_pred = results4.predict(mat_4)[0]

        mat_5_pred = results5.predict(mat_5)[0]

        mat_6_pred = results6.predict(mat_6)[0]

        mat_7_pred = results7.predict(mat_7)[0]

        aluminum_pred = results_aluminum.predict(aluminum)[0]

        corr_cardboard_pred = results_corrcardboard.predict(corr_cardboard)[0]

        food_pred = results_food.predict(food)[0]

        glass_pred = results_glass.predict(glass)[0]

        mixed_paper_pred = results_mixedpaper.predict(mixed_paper)[0]

        newspaper_pred = results_newspaper.predict(newspaper)[0]

        nonrecyclmsw_pred = results_nonrecyclmsw.predict(nonrecyclmsw)[0]

        paperboard_pred = results_paperboard.predict(paperboard)[0]

        soiledpaper_pred = results_soiledpaper.predict(soiledpaper)[0]

        steelferrous_pred = results_steelferrous.predict(steelferrous)[0]

        whitepaper_pred = results_whitepaper.predict(whitepaper)[0]

        yardtrimmings_pred = results_yardtrimmings.predict(yardtrimmings)[0]

        data = {'Material Type': ['#1','#2','#3','#4','#5','#6','#7','Aluminum', 'Corr. Cardboard', 'Food', 'Glass','Mixed Paper', 'Newspaper', 'Non-Recyclable MSW', 'Paperboard', 'Soiled Paper', 'Steel / Ferrous', 'White Paper', 'Yard Trimmings'], 'lbs/day': [mat_1_pred, mat_2_pred, mat_3_pred,mat_4_pred,mat_5_pred,mat_6_pred,mat_7_pred,aluminum_pred, corr_cardboard_pred, food_pred, glass_pred, mixed_paper_pred, newspaper_pred ,nonrecyclmsw_pred, paperboard_pred, soiledpaper_pred, steelferrous_pred, whitepaper_pred, yardtrimmings_pred]}
        datatable = pd.DataFrame(data=data)
        datatable.loc[datatable['lbs/day'].astype(float) < 0,'lbs/day'] = 0
        datatable.sort_values(by=['lbs/day'], inplace = True, ascending = False)
        datatable = datatable.reset_index(drop = True)
        st.table(datatable)


        fig1, ax1 = plt.subplots()
        
        ax1.bar(datatable['Material Type'], datatable['lbs/day'])
        plt.xticks(rotation=90)
        plt.xlabel("Material Type")
        plt.ylabel("lbs/day")

        st.pyplot(fig1)





