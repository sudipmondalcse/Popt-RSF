# Popt-RSF

***Prerequisites :*** <br/>
The required dependencies for the analysis is Python programming ***version 3***
and is tested to work under windows as well as Ubuntu.<br/>


***Dataset preparation*** <br/>
dataset 1 and dataset2 is the input files for the analysis. We have preprocessed the original data after the removal of missing and unspecified values and initialized some labels according to the category. <br/>

*********************

**Label for Dataset1 :** Data Dictionary of the dataset is mentioned below:<br/>
**1. sex:** Patient’s gender (male means 2, female means 1).<br/>
**2. date_symptoms:** Date of the appearance of covid19 symptoms.<br/>
**3. entry_date:** Date of the patient’s first hospital visit.<br/>
**4. date_died:** Date of the patient’s death, “9999-99-99” means the recovery of the patient. <br/>
**5. pneumonia:** Air sacs inflammation of a patient, already existed or not (yes means 1, no means 2). <br/>
**6. age:** Patient’s age. <br/>
**7. diabetes:** The patient is diabetic or not (yes means 1, no means 2). <br/>
**8. asthma:** the patient has asthma or not (yes means 1, no means 2). <br/>
**9. hypertension:** The patient has hypertension or not (yes means 1, no means 2). <br/>
**10. other_disease:** The patient has another disease or not (yes means 1, no means 2). <br/>
**11. renal_chronic:** The patient has renal chronic or not (yes means 1, no means 2). <br/>
**12. tobacco:** If the patient uses tobacco or not (yes means 1, no means 2). <br/>
**13. Duration:** Number of days in between date_symptoms and entry_date. <br/>

Also download from- https://www.kaggle.com/tanmoyx/covid19-patient-precondition-dataset?select=covid.csv
*********************

**Label for Dataset1 :** The Data Dictionary of the dataset is mentioned below: <br/>
**1. Patient Number:** Patient identification number. <br/>
**2. Date Announced:** Date of being covid19 positive. <br/>
**3. Age Bracket:** Patient’s age. <br/>
**4. Gender:** Patient’s gender (male means 2, female means 1) <br/>
**5. Detected State:** State in which case got detected <br/>
**6. Current Status:** Status of the patient, i.e. Hospitalized or Recovered or Died. <br/>
**7. Status Change Date:** Date on which the current status is evaluated.  <br/>
**8. Duration:** Number of days in between Date Announced and Status Change Date. <br/>

*********************

***To start a run :*** <br/>
rsf_default _and_optimized.py for RSF and Popt-RSF
 <br/>



***Feedback*** <br/>
Any further information contact at <br/> 
sudipmondalcs@gmail.com, ananyaghosh15299@gmail.com
