
import numpy as np
from sklearn.svm import SVC as svc
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.linear_model import LogisticRegression as LR
Algorithm = "SVM" or "Decision Tree" or "Logistic Regression"

app = Flask(__name__, template_folder='template')
svc = pickle.load(open('svc_model.pkl','rb'))
DTC = pickle.load(open('LR_model.pkl','rb'))
LR = pickle.load(open('DTC_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')