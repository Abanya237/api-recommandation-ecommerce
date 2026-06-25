from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from surprise import SVD, KNNBasic, Dataset, Reader

app = Flask(__name__)
CORS(app)

# ── Charger les données et entraîner les modèles ─────────────────────────────
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

print(f"✅ Modèles prêts — {matrice['client_id'].nunique()} clients chargés")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return jsonify({
        "message"  : "API Recommandation IA — E-commerce Cameroun",
        "status"   : "actif",
        "clients"  : int(matrice['client_id'].nunique()),
        "produits" : int(matrice['produit'].nunique())
    })

@app.route('/recommander/<client_id>')
def recommander(client_id):
    n = int(request.args.get('n', 5))
    deja_achetes  = matrice[matrice['client_id']==client_id]['produit'].tolist()
    tous_produits = matrice['produit'].unique()
    non_achetes   = [p for p in tous_produits if p not in deja_achetes]

    if not non_achetes:
        return jsonify({
            "client_id"       : client_id,
            "deja_achetes"    : deja_achetes,
            "recommandations" : []
        })

    scores = []
    for produit in non_achetes:
        try:
            s_svd   = svd.predict(client_id, produit).est
            s_knn   = knn.predict(client_id, produit).est
            s_final = round(0.6*s_svd + 0.4*s_knn, 3)
            scores.append({"produit": produit, "score": s_final})
        except:
            pass

    scores.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({
        "client_id"       : client_id,
        "deja_achetes"    : deja_achetes,
        "recommandations" : scores[:n]
    })

@app.route('/tendances')
def tendances():
    n = int(request.args.get('n', 5))
    top = matrice.groupby('produit')['score'].sum()\
                 .sort_values(ascending=False).head(n)
    resultats = [
        {"produit": p, "score": round(float(s), 3), "type": "tendance"}
        for p, s in top.items()
    ]
    return jsonify({
        "type"            : "tendances",
        "description"     : "Produits les plus populaires",
        "recommandations" : resultats
    })

@app.route('/nouveau-client/<produit_vu>')
def nouveau_client(produit_vu):
    n = int(request.args.get('n', 5))
    tous_produits = matrice['produit'].unique()
    non_vus = [p for p in tous_produits if p != produit_vu]

    scores = []
    for produit in non_vus:
        try:
            score = knn.predict("NOUVEAU", produit).est
            scores.append({"produit": produit, "score": round(score, 3)})
        except:
            pass

    scores.sort(key=lambda x: x['score'], reverse=True)
    return jsonify({
        "type"            : "nouveau-client",
        "produit_consulte": produit_vu,
        "recommandations" : scores[:n]
    })

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
