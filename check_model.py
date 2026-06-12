"""Check what attributes are missing from old XGBoost model"""
import joblib, os, xgboost

ML_DIR = r'C:\Users\games\OneDrive\Desktop\PhishRAG資料夾\PhishRAG\PhishRAG_MLPipeline'
m = joblib.load(os.path.join(ML_DIR, 'model_a_binary.pkl'))

print(f"XGBoost version: {xgboost.__version__}")
print(f"Model type: {type(m)}")

# Create a fresh XGBClassifier to see what attributes it has
fresh = xgboost.XGBClassifier()
fresh_attrs = set(vars(fresh).keys())
model_attrs = set(vars(m).keys())

missing = fresh_attrs - model_attrs
extra = model_attrs - fresh_attrs

print(f"\nAttributes in fresh model but MISSING from loaded model:")
for a in sorted(missing):
    print(f"  {a} = {getattr(fresh, a, 'N/A')}")

print(f"\nAttributes in loaded model but not in fresh:")
for a in sorted(extra):
    print(f"  {a} = {getattr(m, a, 'N/A')}")
