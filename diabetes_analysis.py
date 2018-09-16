#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 23:00:27 2018

@author: geoffrey.kip
"""
# Import required packages
from os import chdir
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import pymysql
from pandasql import sqldf
import statsmodels.api as sm
from sklearn.cluster import KMeans

# Set working directory
wd="/Users/geoffrey.kip/Projects/uci_diabetes"
chdir(wd)

# connect to database
host = '127.0.0.1'
user = 'root'
password = ''
port = 3306
db = 'hospitals'

# Connect to sql database with data inside
conn = pymysql.connect(host= host, port=port, user= user, passwd='', db=db)

# Read full dataset and map different id codes
diabetes_df=pd.read_sql("""SELECT * ,
CASE
WHEN admission_type_id=1 THEN "Emergency"
WHEN admission_type_id=2 THEN "Urgent"
WHEN admission_type_id=3 THEN "Elective"
WHEN admission_type_id=4 THEN "Newborn"
WHEN admission_type_id=5 THEN "Not Available"
WHEN admission_type_id=6 THEN "NULL"
WHEN admission_type_id=7 THEN "Trauma Center"
ELSE null
END as admission_type_id_description,
CASE WHEN discharge_disposition_id=1 THEN "Discharged to home"
WHEN discharge_disposition_id=2 THEN "Discharged/transferred to another short term hospital"
WHEN discharge_disposition_id=3 THEN "Discharged/transferred to SNF"
WHEN discharge_disposition_id=4 THEN "Discharged/transferred to ICF"
WHEN discharge_disposition_id=5 THEN "Discharged/transferred to another type of inpatient care institution"
WHEN discharge_disposition_id=6 THEN "Discharged/transferred to home with home health service"
WHEN discharge_disposition_id=7 THEN "Left AMA"
WHEN discharge_disposition_id=8 THEN "Discharged/transferred to home under care of Home IV provider"
WHEN discharge_disposition_id=9 THEN "Admitted as an inpatient to this hospital"
WHEN discharge_disposition_id=10 THEN "Neonate discharged to another hospital for neonatal aftercare"
WHEN discharge_disposition_id=11 THEN "Expired"
WHEN discharge_disposition_id=12 THEN "Still patient or expected to return for outpatient services"
WHEN discharge_disposition_id=13 THEN "Hospice / home"
WHEN discharge_disposition_id=14 THEN "Hospice / medical facility"
WHEN discharge_disposition_id=15 THEN "Discharged/transferred within this institution to Medicare approved swing bed"
WHEN discharge_disposition_id=16 THEN "Discharged/transferred/referred another institution for outpatient services"
WHEN discharge_disposition_id=17 THEN "Discharged/transferred/referred to this institution for outpatient services"
WHEN discharge_disposition_id=18 THEN "NULL"
WHEN discharge_disposition_id=19 THEN "Expired at home. Medicaid only, hospice."
WHEN discharge_disposition_id=20 THEN "Expired in a medical facility. Medicaid only, hospice."
WHEN discharge_disposition_id=21 THEN "Expired, place unknown. Medicaid only, hospice."
WHEN discharge_disposition_id=22 THEN "Discharged/transferred to another rehab fac including rehab units of a hospital ."
WHEN discharge_disposition_id=23 THEN "Discharged/transferred to a long term care hospital."
WHEN discharge_disposition_id=24 THEN "Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare."
WHEN discharge_disposition_id=25 THEN "Not Mapped"
WHEN discharge_disposition_id=26 THEN "Unknown/Invalid"
WHEN discharge_disposition_id=30 THEN "Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere"
WHEN discharge_disposition_id=27 THEN "Discharged/transferred to a federal health care facility"
WHEN discharge_disposition_id=28 THEN "Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital"
WHEN discharge_disposition_id=29 THEN "Discharged/transferred to a Critical Access Hospital (CAH)"
ELSE NULL
END as discharge_disposition_id_description,
CASE
WHEN admission_source_id=1 THEN " Physician Referral"
WHEN admission_source_id=2 THEN "Clinic Referral"
WHEN admission_source_id=3 THEN "HMO Referral"
WHEN admission_source_id=4 THEN "Transfer from a hospital"
WHEN admission_source_id=5 THEN " Transfer from a Skilled Nursing Facility (SNF)"
WHEN admission_source_id=6 THEN " Transfer from another health care facility"
WHEN admission_source_id=7 THEN " Emergency Room"
WHEN admission_source_id=8 THEN " Court/Law Enforcement"
WHEN admission_source_id=9 THEN " Not Available"
WHEN admission_source_id=10 THEN " Transfer from critial access hospital"
WHEN admission_source_id=11 THEN "Normal Delivery"
WHEN admission_source_id=12 THEN " Premature Delivery"
WHEN admission_source_id=13 THEN " Sick Baby"
WHEN admission_source_id=14 THEN " Extramural Birth"
WHEN admission_source_id=15 THEN "Not Available"
WHEN admission_source_id=17 THEN "NULL"
WHEN admission_source_id=18 THEN " Transfer From Another Home Health Agency"
WHEN admission_source_id=19 THEN "Readmission to Same Home Health Agency"
WHEN admission_source_id=20 THEN " Not Mapped"
WHEN admission_source_id=21 THEN "Unknown/Invalid"
WHEN admission_source_id=22 THEN " Transfer from hospital inpt/same fac reslt in a sep claim"
WHEN admission_source_id=23 THEN " Born inside this hospital"
WHEN admission_source_id=24 THEN " Born outside this hospital"
WHEN admission_source_id=25 THEN " Transfer from Ambulatory Surgery Center"
WHEN admission_source_id=26 THEN "Transfer from Hospice"
ELSE NULL
END as admission_source_id_description
FROM hospitals.diabetes;
""", con=conn)

#Basic exploration
diabetes_df.head()
diabetes_df.shape
diabetes_df.describe()
diabetes_df.groupby("readmitted").size()

#Find null or missing
diabetes_df.isnull().sum()
diabetes_df.isna().sum()

#Exploratory analysis time 
# How many encounters by patient
q="""Select 
     patient_nbr,
     count(distinct encounter_id) as encounters
     from diabetes_df
     group by 1
     order by 2 desc"""
patient_encounters = sqldf(q)
       
q="""Select AVG(encounters) as average_encounters from (Select 
     patient_nbr,
     count(distinct encounter_id) as encounters
     from diabetes_df
     group by 1
     order by 2 desc)"""
avg_patient_encounters = sqldf(q)

#Mean figures medications etc
q = """Select 
       AVG(num_procedures) as mean_num_procedures,
       AVG(num_medications) as mean_num_medications,
       AVG(num_lab_procedures) as mean_num_lab_procedures,
       AVG(time_in_hospital) as average_time_in_hospital,
       AVG(number_outpatient) as mean_outpatient_visits,
       AVG(number_emergency) as mean_emergency_visits,
       AVG(number_diagnoses) as mean_number_diagnoses,
       AVG(number_inpatient) as mean_number_inpatient
       from diabetes_df"""
average_measurements = sqldf(q)

qb = """Select 
       race,
       AVG(num_procedures) as mean_num_procedures,
       AVG(num_medications) as mean_num_medications,
       AVG(num_lab_procedures) as mean_num_lab_procedures,
       AVG(time_in_hospital) as average_time_in_hospital,
       AVG(number_outpatient) as mean_outpatient_visits,
       AVG(number_emergency) as mean_emergency_visits,
       AVG(number_diagnoses) as mean_number_diagnoses,
       AVG(number_inpatient) as mean_number_inpatient
       from diabetes_df
       group by 1"""
average_measurements_by_race = sqldf(qb)

qb = """Select 
       readmitted,
       AVG(num_procedures) as mean_num_procedures,
       AVG(num_medications) as mean_num_medications,
       AVG(num_lab_procedures) as mean_num_lab_procedures,
       AVG(time_in_hospital) as average_time_in_hospital,
       AVG(number_outpatient) as mean_outpatient_visits,
       AVG(number_emergency) as mean_emergency_visits,
       AVG(number_diagnoses) as mean_number_diagnoses,
       AVG(number_inpatient) as mean_number_inpatient
       from diabetes_df
       group by 1"""
average_measurements_by_readmission = sqldf(qb)


q0 = """Select weight, 
        count (distinct patient_nbr) as total_individuals, 
        count(distinct (case when readmitted="<30"then patient_nbr else 0 end)) as readmitted_individuals
       from diabetes_df group by 1 
       order by readmitted_individuals desc"""
weight_readmitted_patients = sqldf(q0)
weight_readmitted_patients["percentage_of_individuals_readmitted"]= weight_readmitted_patients['readmitted_individuals']/weight_readmitted_patients['total_individuals'] * 100
weight_readmitted_patients= weight_readmitted_patients.sort_values("percentage_of_individuals_readmitted",ascending=False)
print(weight_readmitted_patients)

# Race distribution of patients against readmission within 30 days
q1 = """Select race, 
        count (distinct patient_nbr) as total_individuals, 
        count(distinct (case when readmitted="<30"then patient_nbr else 0 end)) as readmitted_individuals
       from diabetes_df group by 1 
       order by readmitted_individuals desc"""
race_readmitted_patients = sqldf(q1)
race_readmitted_patients["percentage_of_individuals_readmitted"]= race_readmitted_patients['readmitted_individuals']/race_readmitted_patients['total_individuals'] * 100
race_readmitted_patients= race_readmitted_patients.sort_values("percentage_of_individuals_readmitted",ascending=False)
print(race_readmitted_patients)

#Gender distribution of patients against readmission within 30 days
q2 = """Select gender, 
        count (distinct patient_nbr) as total_individuals, 
        count(distinct (case when readmitted="<30"then patient_nbr else 0 end)) as readmitted_individuals
       from diabetes_df group by 1 
       order by readmitted_individuals desc"""
gender_readmitted_patients = sqldf(q2)
gender_readmitted_patients["percentage_of_individuals_readmitted"]= gender_readmitted_patients['readmitted_individuals']/gender_readmitted_patients['total_individuals'] * 100
gender_readmitted_patients= gender_readmitted_patients.sort_values("percentage_of_individuals_readmitted",ascending=False)
print(gender_readmitted_patients)

# Age distribution of patients against readmission within 30 days
q3 = """Select age, 
        count (distinct patient_nbr) as total_individuals, 
        count(distinct (case when readmitted="<30"then patient_nbr else 0 end)) as readmitted_individuals
       from diabetes_df group by 1 
       order by readmitted_individuals desc"""
age_readmitted_patients = sqldf(q3)
age_readmitted_patients["percentage_of_individuals_readmitted"]= age_readmitted_patients['readmitted_individuals']/age_readmitted_patients['total_individuals'] * 100
age_readmitted_patients= age_readmitted_patients.sort_values("percentage_of_individuals_readmitted",ascending=False)
print(age_readmitted_patients)

# Which variables are correlated with each other?
corr = diabetes_df[diabetes_df.columns.difference(['encounter_id', 'patient_nbr','admission_type_id',
                                                  'discharge_disposition_id','admission_source_id'])].corr()
fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
sns.heatmap(corr, annot = True, ax=ax)

# Is diabetes medication related to readmission within 30 days?
q4 = """Select diabetesMed, 
        count (distinct patient_nbr) as total_individuals, 
        count(distinct (case when readmitted="<30"then patient_nbr else 0 end)) as readmitted_individuals
       from diabetes_df group by 1 
       order by readmitted_individuals desc"""
diabetes_readmitted_patients = sqldf(q4)
diabetes_readmitted_patients["percentage_of_individuals_readmitted"]= diabetes_readmitted_patients['readmitted_individuals']/diabetes_readmitted_patients['total_individuals'] * 100
diabetes_readmitted_patients= diabetes_readmitted_patients.sort_values("percentage_of_individuals_readmitted",ascending=False)
print(diabetes_readmitted_patients)

# Does admission type affect readmission within 30 days?
q5 = """Select admission_type_id_description, 
        count (distinct patient_nbr) as total_individuals, 
        count(distinct (case when readmitted="<30"then patient_nbr else 0 end)) as readmitted_individuals
       from diabetes_df group by 1 
       order by readmitted_individuals desc"""
admission_readmitted_patients = sqldf(q5)
admission_readmitted_patients["percentage_of_individuals_readmitted"]= admission_readmitted_patients['readmitted_individuals']/admission_readmitted_patients['total_individuals'] * 100
admission_readmitted_patients= admission_readmitted_patients.sort_values("percentage_of_individuals_readmitted",ascending=False)
print(admission_readmitted_patients)

# Does discharge type affect readmission within 30 days?
q6= """Select discharge_disposition_id_description, 
        count (distinct patient_nbr) as total_individuals, 
        count(distinct (case when readmitted="<30"then patient_nbr else 0 end)) as readmitted_individuals
       from diabetes_df group by 1 
       order by readmitted_individuals desc"""
discharge_readmitted_patients = sqldf(q6)
discharge_readmitted_patients["percentage_of_individuals_readmitted"]= discharge_readmitted_patients['readmitted_individuals']/discharge_readmitted_patients['total_individuals'] * 100
discharge_readmitted_patients= discharge_readmitted_patients.sort_values("percentage_of_individuals_readmitted",ascending=False)
print(discharge_readmitted_patients)

# Does discharge type affect readmission within 30 days?
q7= """Select admission_source_id_description, 
        count (distinct patient_nbr) as total_individuals, 
        count(distinct (case when readmitted="<30"then patient_nbr else 0 end)) as readmitted_individuals
       from diabetes_df group by 1 
       order by readmitted_individuals desc"""
admission_source_readmitted_patients = sqldf(q7)
admission_source_readmitted_patients["percentage_of_individuals_readmitted"]= admission_source_readmitted_patients['readmitted_individuals']/admission_source_readmitted_patients['total_individuals'] * 100
admission_source_readmitted_patients= admission_source_readmitted_patients.sort_values("percentage_of_individuals_readmitted",ascending=False)
print(admission_source_readmitted_patients)

# Does discharge type affect readmission within 30 days?
q8= """Select medical_specialty, 
        count (distinct patient_nbr) as total_individuals, 
        count(distinct (case when readmitted="<30"then patient_nbr else 0 end)) as readmitted_individuals
       from diabetes_df group by 1 
       order by readmitted_individuals desc"""
medical_speciality_readmitted_patients = sqldf(q8)
medical_speciality_readmitted_patients["percentage_of_individuals_readmitted"]= medical_speciality_readmitted_patients['readmitted_individuals']/medical_speciality_readmitted_patients['total_individuals'] * 100
medical_speciality_readmitted_patients= medical_speciality_readmitted_patients.sort_values("percentage_of_individuals_readmitted",ascending=False)
print(medical_speciality_readmitted_patients)

# Does A1cresult affect readmission within 30 days
q9= """Select A1Cresult, 
        count (distinct patient_nbr) as total_individuals, 
        count(distinct (case when readmitted="<30"then patient_nbr else 0 end)) as readmitted_individuals
       from diabetes_df group by 1 
       order by readmitted_individuals desc"""
a1c_readmitted_patients = sqldf(q9)
a1c_readmitted_patients["percentage_of_individuals_readmitted"]= a1c_readmitted_patients['readmitted_individuals']/a1c_readmitted_patients['total_individuals'] * 100
a1c_readmitted_patients= a1c_readmitted_patients.sort_values("percentage_of_individuals_readmitted",ascending=False)
print(a1c_readmitted_patients)

# Did change in medications lead to readmission?
q10= """Select any_change, 
        count (distinct patient_nbr) as total_individuals, 
        count(distinct (case when readmitted="<30"then patient_nbr else 0 end)) as readmitted_individuals
       from diabetes_df group by 1 
       order by readmitted_individuals desc"""
change_readmitted_patients = sqldf(q10)
change_readmitted_patients["percentage_of_individuals_readmitted"]= change_readmitted_patients['readmitted_individuals']/change_readmitted_patients['total_individuals'] * 100
change_readmitted_patients= change_readmitted_patients.sort_values("percentage_of_individuals_readmitted",ascending=False)
print(change_readmitted_patients)

# Did glucose serum levels lead to readmission?
q11= """Select max_glu_serum, 
        count (distinct patient_nbr) as total_individuals, 
        count(distinct (case when readmitted="<30"then patient_nbr else 0 end)) as readmitted_individuals
       from diabetes_df group by 1 
       order by readmitted_individuals desc"""
gluserum_readmitted_patients = sqldf(q11)
gluserum_readmitted_patients["percentage_of_individuals_readmitted"]= gluserum_readmitted_patients['readmitted_individuals']/gluserum_readmitted_patients['total_individuals'] * 100
gluserum_readmitted_patients= gluserum_readmitted_patients.sort_values("percentage_of_individuals_readmitted",ascending=False)
print(gluserum_readmitted_patients)

# Look at all different medications
medications=['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
       'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'examide', 'citoglipton', 'insulin',
       'glyburide_metformin', 'glipizide_metformin',
       'glimepiride_pioglitazone', 'metformin_rosiglitazone',
       'metformin_pioglitazone']

diabetes_df["medications_status"] = np.where(diabetes_df[medications][diabetes_df == "Steady"].any(1), 'steady', 
                             np.where(diabetes_df[medications][diabetes_df == "Up"].any(1), 'up' ,
                             np.where(diabetes_df[medications][diabetes_df == "Down"].any(1), 'down',
                             np.where(diabetes_df[medications][diabetes_df == "No"].any(1), 'not_prescribed','unknown'))))

# =============================================================================
# medications1=['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
#        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
#        'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
#        'tolazamide', 'examide', 'citoglipton', 'insulin',
#        'glyburide_metformin', 'glipizide_metformin',
#        'glimepiride_pioglitazone', 'metformin_rosiglitazone',
#        'metformin_pioglitazone','medications_status']
# 
# check= diabetes_df[medications1]
# =============================================================================

# Did medications_status lead to readmission?
q12= """Select medications_status, 
        count (distinct patient_nbr) as total_individuals, 
        count(distinct (case when readmitted="<30"then patient_nbr else 0 end)) as readmitted_individuals
       from diabetes_df group by 1 
       order by readmitted_individuals desc"""
medication_status_readmitted_patients = sqldf(q12)
medication_status_readmitted_patients["percentage_of_individuals_readmitted"]= medication_status_readmitted_patients['readmitted_individuals']/medication_status_readmitted_patients['total_individuals'] * 100
medication_status_readmitted_patients= medication_status_readmitted_patients.sort_values("percentage_of_individuals_readmitted",ascending=False)
print(medication_status_readmitted_patients)

# Do some clustering and modeling
# simple statical regression model to understand beta coefficients for some data points
train_cols=['num_procedures', 'num_medications','num_lab_procedures','time_in_hospital','number_outpatient',
               'number_emergency','number_diagnoses','number_inpatient']
# Recode readmitted to binary for model
diabetes_df["readmitted"]= np.where(diabetes_df["readmitted"] == '<30',1,0)
logit = sm.Logit(diabetes_df['readmitted'], diabetes_df[train_cols])

# fit the model
result = logit.fit()
print (result.summary2())

# Clustering kmeans on same features
# Initializing KMeans
kmeans = KMeans(n_clusters=4)
# Fitting with inputs
kmeans = kmeans.fit(diabetes_df[train_cols])
# Predicting the clusters
labels = kmeans.predict(diabetes_df[train_cols])
# Getting the cluster centers
C = kmeans.cluster_centers_
X=diabetes_df[train_cols]
y=diabetes_df["readmitted"]

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X.iloc[:, 0], X.iloc[:, 1], X.iloc[:, 2], c=y)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)
