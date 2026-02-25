import random

TIFINAGH_CHARS = [
  'ⴰ','ⴱ','ⵛ','ⴷ','ⴻ','ⴼ','ⴳ',
    'ⵀ','ⵉ','ⵊ','ⴽ','ⵍ','ⵎ','ⵏ','ⵇ',
    'ⵔ','ⵙ','ⵜ','ⵓ','ⵡ','ⵢ','ⵅ','ⵣ','ⵃ','ⵚ','ⴹ','ⵟ',
    'ⵄ','ⵖ','ⵥ','ⴳⵯ','ⴽⵯ','ⵕ',
]

DICTIONNAIRE_TIFINAGH = [
    "ⴰⵎⴰⵣⵉⵖ", "ⵜⴰⵎⴰⵣⵉⵖⵜ", "ⴰⴼⵍⵍⴰ", "ⵜⴰⴼⵓⴽⵜ", "ⴰⵎⴰⵏ",
    "ⴰⴷⵔⴰⵔ", "ⵜⴰⵎⵓⵔⵜ", "ⴰⵙⵉⴼ", "ⵉⵣⵍⵉ", "ⵜⴰⵙⴽⵍⴰ",
    "ⴰⵏⴰⴼ", "ⵜⴰⵏⴰⴼⵜ", "ⵉⵎⵉ", "ⵜⵉⵔⵔⴰ", "ⴰⵙⵙ",
    "ⵉⴹ", "ⴰⵢⵢⵓⵔ", "ⵉⵜⵔⵉ", "ⵜⴰⵙⵓⵜ", "ⴰⴳⴰⵔ",
    "ⵜⵉⵖⵔⵉ", "ⴰⵙⵙⴰⵖ", "ⵜⵉⵙⵙⵉ", "ⴰⵎⵓⵔ", "ⵉⵏⵏⴰ",
    "ⵜⴻⵏⵏⴰ", "ⵉⵡⵡⴻⵜ", "ⴰⵔⵓ", "ⵉⵔⴰ", "ⵜⴰⵡⴰⵍⴰ"
]


def levenshtein(s1, s2):
    """Calcule la distance d'édition entre deux chaînes."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]


def corriger_mot(mot_errone, dictionnaire=None, top_k=3):
    """Retourne le mot le plus proche dans le dictionnaire."""
    if dictionnaire is None:
        dictionnaire = DICTIONNAIRE_TIFINAGH
    distances = sorted([(mot, levenshtein(mot_errone, mot)) for mot in dictionnaire],
                       key=lambda x: x[1])
    return distances[0][0], distances[:top_k]


def simuler_erreur_ocr(mot, taux_erreur=0.2):
    """Simule une erreur OCR en remplaçant aléatoirement des caractères."""
    mot_liste = list(mot)
    nb_erreurs = max(1, int(len(mot) * taux_erreur))
    for _ in range(nb_erreurs):
        idx = random.randint(0, len(mot_liste) - 1)
        mot_liste[idx] = random.choice([c for c in TIFINAGH_CHARS if c != mot_liste[idx]])
    return ''.join(mot_liste)


def evaluer_correcteur(dictionnaire=None, n_tests=10, taux_erreur=0.25, seed=42):
    """Évalue le module de correction et retourne les métriques."""
    if dictionnaire is None:
        dictionnaire = DICTIONNAIRE_TIFINAGH
    random.seed(seed)
    mots_test = random.sample(dictionnaire, min(n_tests, len(dictionnaire)))
    corrections_reussies = 0
    resultats = []
    for mot_original in mots_test:
        mot_errone = simuler_erreur_ocr(mot_original, taux_erreur)
        mot_corrige, _ = corriger_mot(mot_errone, dictionnaire)
        succes = (mot_corrige == mot_original)
        if succes: corrections_reussies += 1
        resultats.append({'original': mot_original, 'errone': mot_errone,
                          'corrige': mot_corrige, 'succes': succes})
    taux = corrections_reussies / len(mots_test) * 100
    wer_avant = sum(1 for r in resultats if r['errone']  != r['original']) / len(resultats)
    wer_apres = sum(1 for r in resultats if r['corrige'] != r['original']) / len(resultats)
    return resultats, taux, wer_avant * 100, wer_apres * 100