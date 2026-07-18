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

# ============================================================
# NOUVEAU : matrice de co-occurrence produit-produit
# Sert à répondre à /nouveau-client/<produit_vu> avec de VRAIES
# similarités, au lieu d'interroger un utilisateur fictif "NOUVEAU"
# qui renvoyait toujours la moyenne globale (bug corrigé ici).
# ============================================================
print("⏳ Construction de la matrice de co-occurrence produit-produit...")
panier_par_client = commandes.groupby('client_id')['produit_court'].apply(set)

from collections import defaultdict
co_occurrence = defaultdict(lambda: defaultdict(int))
for produits_achetes in panier_par_client:
    produits_achetes = list(produits_achetes)
    for i in range(len(produits_achetes)):
        for j in range(len(produits_achetes)):
            if i != j:
                co_occurrence[produits_achetes[i]][produits_achetes[j]] += 1

def produits_similaires(produit_vu, n=5):
    """Retourne les n produits les plus souvent achetés avec produit_vu."""
    if produit_vu not in co_occurrence or len(co_occurrence[produit_vu]) == 0:
        # Repli : aucune co-occurrence connue -> tendances générales
        top = matrice.groupby('produit')['score'].sum().sort_values(ascending=False)
        top = top[top.index != produit_vu].head(n)
        return [{"produit": p, "score": round(float(s), 3)} for p, s in top.items()]

    voisins = co_occurrence[produit_vu]
    tries = sorted(voisins.items(), key=lambda x: x[1], reverse=True)[:n]
    max_count = tries[0][1] if tries else 1
    return [{"produit": p, "score": round(count / max_count, 3)} for p, count in tries]

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
    """
    CORRIGÉ : utilise désormais la co-occurrence produit-produit
    (quels produits sont achetés ensemble) plutôt que knn.predict("NOUVEAU", p),
    qui renvoyait toujours la même moyenne globale pour tout utilisateur inconnu.
    """
    n = int(request.args.get('n', 5))
    recos = produits_similaires(produit_vu, n)
    return jsonify({
        "type": "nouveau-client",
        "produit_consulte": produit_vu,
        "recommandations": recos
    })


@app.route('/health')
def health():
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
