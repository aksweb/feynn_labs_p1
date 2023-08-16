from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)

# Load the model from the pickle file
model = pickle.load(open('model.pkl', 'rb'))


sympp_list= [' throat_irritation',
 ' anxiety',
 ' muscle_weakness',
 ' toxic_look_(typhos)',
 ' irregular_sugar_level',
 ' fatigue',
 ' receiving_unsterile_injections',
 ' blood_in_sputum',
 ' pain_in_anal_region',
 ' cold_hands_and_feets',
 ' receiving_blood_transfusion',
 ' blackheads',
 ' red_spots_over_body',
 ' spotting_ urination',
 ' distention_of_abdomen',
 ' loss_of_appetite',
 ' slurred_speech',
 ' lethargy',
 ' weakness_in_limbs',
 ' continuous_sneezing',
 ' watering_from_eyes',
 ' spinning_movements',
 ' joint_pain',
 ' fluid_overload',
 ' extra_marital_contacts',
 ' loss_of_balance',
 ' bloody_stool',
 ' foul_smell_of urine',
 ' inflammatory_nails',
 ' swollen_legs',
 'itching',
 ' yellowing_of_eyes',
 ' belly_pain',
 ' palpitations',
 ' chest_pain',
 ' visual_disturbances',
 ' movement_stiffness',
 ' irritation_in_anus',
 ' bladder_discomfort',
 ' burning_micturition',
 ' blister',
 ' red_sore_around_nose',
 ' rusty_sputum',
 ' stomach_pain',
 ' sweating',
 ' abnormal_menstruation',
 ' unsteadiness',
 ' passage_of_gases',
 ' mood_swings',
 ' enlarged_thyroid',
 ' stiff_neck',
 ' ulcers_on_tongue',
 ' altered_sensorium',
 ' yellow_urine',
 ' muscle_wasting',
 ' bruising',
 ' weakness_of_one_body_side',
 ' weight_loss',
 ' sinus_pressure',
 ' weight_gain',
 ' swelling_of_stomach',
 ' phlegm',
 ' diarrhoea',
 ' pus_filled_pimples',
 ' painful_walking',
 ' yellow_crust_ooze',
 ' runny_nose',
 ' cramps',
 ' puffy_face_and_eyes',
 ' redness_of_eyes',
 ' knee_pain',
 ' yellowish_skin',
 ' excessive_hunger',
 ' skin_rash',
 ' obesity',
 ' family_history',
 ' nodal_skin_eruptions',
 ' sunken_eyes',
 ' hip_joint_pain',
 ' skin_peeling',
 ' internal_itching',
 ' nausea',
 ' restlessness',
 ' irritability',
 ' fast_heart_rate',
 ' prominent_veins_on_calf',
 ' history_of_alcohol_consumption',
 ' dark_urine',
 ' dischromic _patches',
 ' blurred_and_distorted_vision',
 ' depression',
 ' mild_fever',
 ' continuous_feel_of_urine',
 ' swelled_lymph_nodes',
 ' constipation',
 ' swollen_blood_vessels',
 ' increased_appetite',
 ' coma',
 ' congestion',
 ' acidity',
 ' dizziness',
 ' breathlessness',
 ' dehydration',
 ' stomach_bleeding',
 ' mucoid_sputum',
 ' swelling_joints',
 ' cough',
 ' swollen_extremeties',
 ' malaise',
 ' scurring',
 ' neck_pain',
 ' high_fever',
 ' small_dents_in_nails',
 ' chills',
 ' indigestion',
 ' muscle_pain',
 ' pain_during_bowel_movements',
 ' headache',
 ' silver_like_dusting',
 ' brittle_nails',
 ' drying_and_tingling_lips',
 ' patches_in_throat',
 ' acute_liver_failure',
 ' polyuria',
 ' shivering',
 ' vomiting',
 ' abdominal_pain',
 ' back_pain',
 ' loss_of_smell',
 ' lack_of_concentration',
 ' pain_behind_the_eyes']
symp_list = []
for item in sympp_list:
    symp_list.append(item.lstrip())
#indexing
ind_dict={}
for index, item in enumerate(symp_list):
    ind_dict[item] = index


@app.route('/')
def home():
    return render_template('index.html')
inp=[0] * 131
@app.route('/predict', methods=['POST'])
def predict():
    # return "Thank You"
    selected_symptoms = request.form.getlist('symptoms')

    # Set positions in the zero_list to 1 based on string values
    for st in selected_symptoms:
     val = ind_dict[st]
     inp[int(val+1)] = 1
    input_array = np.array(inp)
    # print(input_array)


    # numbered_symptoms = {symptom: index for index, symptom in enumerate(selected_symptoms)}

    # Perform prediction using the loaded model
    prediction = model.predict(input_array.reshape(1,131))  # Replace with the actual
    # prediction code

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
