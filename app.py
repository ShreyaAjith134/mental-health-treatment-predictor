from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

with open("Stacking.pkl", "rb") as f:
    model = pickle.load(f)

# Automatically get the exact column order the model was trained with
FEATURE_COLUMNS = model.feature_names_in_.tolist()
print("✅ Model expects these columns in order:")
print(FEATURE_COLUMNS)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect all form values
        data = {col: request.form.get(col, 0) for col in FEATURE_COLUMNS}
        input_df = pd.DataFrame([data])
        input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Reorder columns exactly as model expects
        input_df = input_df[FEATURE_COLUMNS]

        prediction = model.predict(input_df)[0]
        result = "⚠️ Likely needs mental health treatment" if prediction == 1 \
                 else "✅ No immediate treatment indicated"
        return jsonify({"result": result})
    except Exception as ex:
        return jsonify({"result": f"❌ Error: {str(ex)}"})

if __name__ == '__main__':
    app.run(debug=True)