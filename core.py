"""
Core of the "same recipe, different weights" experiment.
Shared by the command-line script (run_experiment.py) and the PyQt GUI (app.py).

Dataset : scikit-learn digits (bundled, no download; 1,797 images, 10 classes)
Model   : MLP 64-128-10 (scikit-learn MLPClassifier), trained from scratch

Experiments
  A. Human choice changes function : vary hidden size and learning rate, seed fixed
  B. Same choice, different arrays  : fix recipe, vary seed (5 runs)
  C. Same function, different array : permute hidden units of one model
"""
import itertools, warnings
from dataclasses import dataclass
import numpy as np
warnings.filterwarnings("ignore")
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

DATA_SPLIT_SEED = 0          # data split is fixed for every run
HIDDEN = 128
BASE = dict(hidden_layer_sizes=(HIDDEN,), solver="adam", batch_size=64,
            max_iter=200, tol=1e-6, n_iter_no_change=200)

A_HIDDEN = [4, 16, 128]
A_LR = [1e-1, 1e-3, 1e-5]
A_EPOCHS = 50
B_LR = 0.001
B_SEEDS = 5
C_PERM_SEED = 123


@dataclass
class Data:
    Xtr: np.ndarray
    Xte: np.ndarray
    ytr: np.ndarray
    yte: np.ndarray


def load_data():
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=DATA_SPLIT_SEED, stratify=y)
    sc = StandardScaler().fit(Xtr)
    return Data(sc.transform(Xtr), sc.transform(Xte), ytr, yte)


def make_model(hidden, lr, seed, epochs):
    return MLPClassifier(hidden_layer_sizes=(hidden,), learning_rate_init=lr, random_state=seed,
                         solver="adam", batch_size=64, max_iter=epochs, tol=1e-6, n_iter_no_change=200)


def train(data, lr, seed):
    m = MLPClassifier(learning_rate_init=lr, random_state=seed, **BASE)
    m.fit(data.Xtr, data.ytr)
    return m


def flat(m):
    return np.concatenate([w.ravel() for w in m.coefs_] + [b.ravel() for b in m.intercepts_])


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


def permute(m, perm):
    """Return weight vector of m with hidden units reordered by perm (function unchanged)."""
    W1, W2 = m.coefs_
    b1, b2 = m.intercepts_
    return np.concatenate([W1[:, perm].ravel(), W2[perm, :].ravel(), b1[perm].ravel(), b2.ravel()])


def align_perm(ref, m):
    """Best permutation of m's hidden units to match ref (Hungarian on incoming+outgoing weights)."""
    f_ref = np.hstack([ref.coefs_[0].T, ref.intercepts_[0][:, None], ref.coefs_[1]])
    f_m = np.hstack([m.coefs_[0].T, m.intercepts_[0][:, None], m.coefs_[1]])
    C = -(f_ref / np.linalg.norm(f_ref, axis=1, keepdims=True)) @ (f_m / np.linalg.norm(f_m, axis=1, keepdims=True)).T
    _, col = linear_sum_assignment(C)
    return col


def align(ref, m):
    return permute(m, align_perm(ref, m))


def rescale(m):
    """Remove the ReLU rescaling symmetry: scale each hidden unit so that its incoming weights and bias
    have unit norm, and scale its outgoing weights inversely. Since relu(a*z) = a*relu(z) for a > 0,
    the function is unchanged."""
    W1, W2 = m.coefs_
    b1, b2 = m.intercepts_
    s = np.sqrt((W1 ** 2).sum(0) + b1 ** 2)
    return W1 / s, b1 / s, W2 * s[:, None], b2


def flat_rescaled(m, perm=None, first_layer=False):
    """Weight vector of the rescaled model (same ordering as flat), hidden units reordered by perm."""
    W1, b1, W2, b2 = rescale(m)
    if perm is None:
        perm = np.arange(W1.shape[1])
    if first_layer:
        return np.concatenate([W1[:, perm].ravel(), b1[perm]])
    return np.concatenate([W1[:, perm].ravel(), W2[perm, :].ravel(), b1[perm], b2])


def forward(m, X, perm=None):
    """Logits of a one-hidden-layer ReLU MLP, optionally with hidden units reordered."""
    W1, W2 = m.coefs_
    b1, b2 = m.intercepts_
    if perm is None:
        perm = np.arange(W1.shape[1])
    h = np.maximum(0, X @ W1[:, perm] + b1[perm])
    return h @ W2[perm, :] + b2


def log10_factorial(n):
    return float(np.sum(np.log10(np.arange(1, n + 1))))


def _noop(pct, msg):
    pass


# ---- Experiment A : same seed, same training budget (50 epochs); vary the human choices
def run_A(data, progress=_noop):
    A = []
    total = len(A_HIDDEN) * len(A_LR)
    for k, (hidden, lr) in enumerate(itertools.product(A_HIDDEN, A_LR)):
        progress(int(30 * k / total), f"실험 A: 은닉 {hidden}, 학습률 {lr:g} 학습 중")
        m = make_model(hidden, lr, 0, A_EPOCHS)
        m.fit(data.Xtr, data.ytr)
        A.append({"hidden_units": hidden, "lr": lr, "seed": 0, "epochs": A_EPOCHS,
                  "test_acc": round(m.score(data.Xte, data.yte), 4)})
    return A


# ---- Experiment B : same recipe, 5 seeds
def run_B(data, progress=_noop):
    models = []
    for s in range(B_SEEDS):
        progress(30 + int(55 * s / B_SEEDS), f"실험 B: seed {s} 학습 중 (200 epochs)")
        models.append(train(data, B_LR, s))
    progress(86, "실험 B: 가중치 비교 및 유닛 정렬")
    accs = [m.score(data.Xte, data.yte) for m in models]
    preds = [m.predict(data.Xte) for m in models]
    pairs = list(itertools.combinations(range(B_SEEDS), 2))
    perms = {(i, j): align_perm(models[i], models[j]) for i, j in pairs}
    raw_cos = [cos(flat(models[i]), flat(models[j])) for i, j in pairs]
    aligned_cos = [cos(flat(models[i]), permute(models[j], perms[i, j])) for i, j in pairs]
    rescaled_cos = [cos(flat_rescaled(models[i]), flat_rescaled(models[j], perms[i, j])) for i, j in pairs]
    rescaled_l1_cos = [cos(flat_rescaled(models[i], first_layer=True),
                           flat_rescaled(models[j], perms[i, j], first_layer=True)) for i, j in pairs]
    raw_l2 = [rel_l2(flat(models[i]), flat(models[j])) for i, j in pairs]
    agree = [float((preds[i] == preds[j]).mean()) for i, j in pairs]

    def rng(v):
        return [round(min(v), 4), round(max(v), 4)]

    summary = {
        "lr": B_LR,
        "test_acc_each": [round(a, 4) for a in accs],
        "test_acc_mean": round(float(np.mean(accs)), 4),
        "test_acc_sd": round(float(np.std(accs, ddof=1)), 4),
        "prediction_agreement_mean": round(float(np.mean(agree)), 4),
        "weight_cosine_raw_mean": round(float(np.mean(raw_cos)), 4),
        "weight_cosine_raw_range": rng(raw_cos),
        "weight_cosine_after_alignment_mean": round(float(np.mean(aligned_cos)), 4),
        "weight_cosine_after_alignment_range": rng(aligned_cos),
        "weight_cosine_after_alignment_and_rescaling_mean": round(float(np.mean(rescaled_cos)), 4),
        "weight_cosine_after_alignment_and_rescaling_range": rng(rescaled_cos),
        "first_layer_cosine_after_alignment_and_rescaling_mean": round(float(np.mean(rescaled_l1_cos)), 4),
        "weight_rel_l2_raw_mean": round(float(np.mean(raw_l2)), 4),
    }

    def matrix(vals):
        M = np.eye(B_SEEDS)
        for (i, j), v in zip(pairs, vals):
            M[i, j] = M[j, i] = v
        return M

    details = {"models": models, "agree": matrix(agree), "raw_cos": matrix(raw_cos),
               "aligned_cos": matrix(aligned_cos), "rescaled_cos": matrix(rescaled_cos),
               "perm_1_to_0": perms[0, 1]}
    return summary, details


# ---- Experiment C : permute the hidden units of one trained model
def permutation_check(m, data, perm):
    logits_orig = forward(m, data.Xte)
    logits_perm = forward(m, data.Xte, perm)
    pred_perm = m.classes_[logits_perm.argmax(1)]
    return {
        "prediction_identical": bool((pred_perm == m.predict(data.Xte)).all()),
        "weight_cosine_original_vs_permuted": round(cos(flat(m), permute(m, perm)), 4),
        "logits_orig": logits_orig,
        "logits_perm": logits_perm,
        "max_abs_logit_diff": float(np.abs(logits_orig - logits_perm).max()),
    }


def run_C(m0, data, seed=C_PERM_SEED):
    perm = np.random.default_rng(seed).permutation(HIDDEN)
    res = permutation_check(m0, data, perm)
    summary = {
        "prediction_identical": res["prediction_identical"],
        "weight_cosine_original_vs_permuted": res["weight_cosine_original_vs_permuted"],
        "number_of_equivalent_arrays": f"{HIDDEN}! (hidden-unit orderings) ≈ 10^{int(log10_factorial(HIDDEN))}",
    }
    return summary, dict(res, perm=perm)


def run_all(progress=_noop):
    """Run A, B, C. Returns (results dict identical to results.json, details for plotting)."""
    data = load_data()
    out = {"dataset": "sklearn digits", "model": f"MLP 64-{HIDDEN}-10", "base": BASE}
    out["A_human_choices"] = run_A(data, progress)
    out["B_seed"], dB = run_B(data, progress)
    progress(95, "실험 C: 은닉 유닛 순서 섞기")
    out["C_permutation"], dC = run_C(dB["models"][0], data)
    progress(100, "완료")
    return out, {"data": data, "B": dB, "C": dC}


def train_custom(hidden, lr, seed, epochs, progress=_noop):
    """Train one model with user-chosen settings (GUI 'try it yourself' panel)."""
    progress(0, f"은닉 {hidden}, 학습률 {lr:g}, seed {seed}, {epochs} epochs 학습 중")
    data = load_data()
    m = make_model(hidden, lr, seed, epochs)
    m.fit(data.Xtr, data.ytr)
    progress(100, "완료")
    return {"hidden": hidden, "lr": lr, "seed": seed, "epochs": epochs,
            "test_acc": m.score(data.Xte, data.yte), "loss_curve": list(m.loss_curve_)}
