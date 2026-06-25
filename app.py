from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import json, re
from surprise import SVD, KNNBasic, Dataset, Reader

app = Flask(__name__)
CORS(app)

print("⏳ Chargement des données...")
commandes = pd.read_csv("commandes_fusionnees.csv")

matrice = commandes.groupby(
    ['client_id','produit_court'])['quantite'].sum().reset_index()
matrice.columns = ['client_id','produit','score']
matrice['score'] = matrice['score'].clip(upper=5)

reader   = Reader(rating_scale=(1, 5))
dataset  = Dataset.load_from_df(matrice[['client_id','produit','score']], reader)
trainset = dataset.build_full_trainset()

svd = SVD(n_factors=50, n_epochs=20, random_state=42)
svd.fit(trainset)

knn = KNNBasic(k=20, sim_options={'name':'cosine','user_based':False})
knn.fit(trainset)

print("⏳ Chargement table de correspondance...")
try:
    with open("tel_client_map.json", "r") as f:
        tel_map = json.load(f)
    print(f"✅ {len(tel_map)} correspondances chargées")
except Exception as e:
    tel_map = {}
    print(f"⚠️ Erreur chargement JSON : {e}")

print(f"✅ Modèles prêts — {matrice['client_id'].nunique()} clients")

@app.route('/')
def home():
    return jsonify({
        "message"        : "API Recommandation IA — E-commerce Cameroun",
        "status"         : "actif",
        "clients"        : int(matrice['client_id'].nunique()),
        "produits"       : int(matrice['produit'].nunique()),
        "correspondances": len(tel_map)
    })

@app.route('/identifier')
def identifier():
    telephone = request.args.get('tel', '')
    tel_clean = re.sub(r'\D', '', telephone)[-9:]
    if tel_clean in tel_map:
        client_id = tel_map[tel_clean]
        nb_achats = int(matrice[matrice['client_id']==client_id]['score'].count())
        return jsonify({"trouve": True, "client_id": client_id, "nb_achats": nb_achats})
    return jsonify({"trouve": False, "client_id": None,
                    "message": "Nouveau client — recommandations génériques"})

@app.route('/recommander/<client_id>')
def recommander(client_id):
    n = int(request.args.get('n', 5))
    deja   = matrice[matrice['client_id']==client_id]['produit'].tolist()
    tous   = matrice['produit'].unique()
    non_a  = [p for p in tous if p not in deja]
    if not non_a:
        return jsonify({"client_id": client_id, "deja_achetes": deja, "recommandations": []})
    scores = []
    for p in non_a:
        try:
            s = round(0.6*svd.predict(client_id,p).est + 0.4*knn.predict(client_id,p).est, 3)
            scores.append({"produit": p, "score": s})
        except: pass
    scores.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({"client_id": client_id, "deja_achetes": deja, "recommandations": scores[:n]})

@app.route('/tendances')
def tendances():
    n   = int(request.args.get('n', 5))
    top = matrice.groupby('produit')['score'].sum().sort_values(ascending=False).head(n)
    return jsonify({
        "type": "tendances",
        "recommandations": [{"produit": p, "score": round(float(s),3), "type":"tendance"}
                             for p,s in top.items()]
    })

@app.route('/nouveau-client/<produit_vu>')
def nouveau_client(produit_vu):
    n    = int(request.args.get('n', 5))
    tous = matrice['produit'].unique()
    scores = []
    for p in [x for x in tous if x != produit_vu]:
        try:
            scores.append({"produit": p, "score": round(knn.predict("NOUVEAU",p).est,3)})
        except: pass
    scores.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({"type":"nouveau-client","produit_consulte":produit_vu,"recommandations":scores[:n]})

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
