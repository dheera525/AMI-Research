"use client";

import { useEffect, useMemo, useState } from "react";

// When deployed, the FastAPI backend serves this static build from the same
// origin, so relative URLs just work. For local dev point this at the backend
// by setting NEXT_PUBLIC_API_URL=http://localhost:8000 before `npm run dev`.
const API_BASE = (process.env.NEXT_PUBLIC_API_URL || "") + "/api";

type FeatureRange = {
  min: number;
  max: number;
  median: number;
  mean: number;
  std: number;
};

type FeaturesResp = {
  features: string[];
  ranges: Record<string, FeatureRange>;
};

type ModelMetric = {
  name: string;
  accuracy: number;
  precision: number;
  sensitivity: number;
  specificity: number;
  f1: number;
  auc_roc: number;
};

type MetricsResp = {
  winner: string;
  models: ModelMetric[];
  all_models?: ModelMetric[];
  n_train: number;
  n_test: number;
  n_features: number;
  top_k_served: number;
};

type PredictionResp = {
  winner: string;
  probability: number;
  prediction: number;
  verdict: string;
  all_models: { model: string; probability: number; prediction: number }[];
};

// ---------------------------------------------------------------------------
// Group features into clinical sections by simple keyword matching.
// Anything that doesn't match a known group ends up in "Other".
// ---------------------------------------------------------------------------
const SECTION_RULES: { name: string; match: (f: string) => boolean }[] = [
  {
    name: "Demographics",
    match: (f) =>
      /\b(age|sex|gender|bmi|weight|height|smok|diab|hypert)\b/i.test(f),
  },
  {
    name: "Complete Blood Count",
    match: (f) =>
      /\b(wbc|rbc|hgb|hb|hct|mcv|mch|mchc|rdw|plt|mpv|pdw|neu|ly|mo|eo|ba|plat|lymph|neutro|mono|eosin|baso)\b/i.test(
        f
      ) || /\//.test(f),
  },
  {
    name: "Lipid Panel",
    match: (f) => /\b(ldl|hdl|tg|trig|chol|lipo)\b/i.test(f),
  },
  {
    name: "Cardiac Markers",
    match: (f) =>
      /\b(troponin|trop|ck|ck-?mb|bnp|nt-?probnp|myoglobin|d-?dimer)\b/i.test(
        f
      ),
  },
  {
    name: "Metabolic & Renal",
    match: (f) =>
      /\b(glu|gluco|hba1c|crea|urea|bun|gfr|na|k|cl|ca|mg|alt|ast|alp|bili|ldh|crp|esr)\b/i.test(
        f
      ),
  },
];

function groupFeatures(features: string[]) {
  const groups: Record<string, string[]> = {};
  const order: string[] = [];
  for (const rule of SECTION_RULES) order.push(rule.name);
  order.push("Other");
  for (const n of order) groups[n] = [];

  for (const f of features) {
    const rule = SECTION_RULES.find((r) => r.match(f));
    groups[rule ? rule.name : "Other"].push(f);
  }
  // Drop empty sections
  return order
    .map((n) => ({ name: n, features: groups[n] }))
    .filter((g) => g.features.length > 0);
}

// ---------------------------------------------------------------------------
// Confidence ring (pure SVG)
// ---------------------------------------------------------------------------
function ConfidenceRing({
  value,
  positive,
}: {
  value: number; // 0..1
  positive: boolean;
}) {
  const size = 130;
  const stroke = 12;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const dash = c * value;
  const color = positive ? "#dc2626" : "#059669";
  return (
    <div className="ring-wrap">
      <svg width={size} height={size}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="#e5edf5"
          strokeWidth={stroke}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeDasharray={`${dash} ${c}`}
          strokeLinecap="round"
        />
      </svg>
      <div className="ring-text">
        <div>
          <div className="pct">{(value * 100).toFixed(1)}%</div>
          <div className="lbl">AMI prob.</div>
        </div>
      </div>
    </div>
  );
}

export default function Home() {
  const [features, setFeatures] = useState<FeaturesResp | null>(null);
  const [metrics, setMetrics] = useState<MetricsResp | null>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResp | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API_BASE}/features`).then((r) => r.json()),
      fetch(`${API_BASE}/metrics`).then((r) => r.json()),
    ])
      .then(([f, m]) => {
        setFeatures(f);
        setMetrics(m);
        const init: Record<string, string> = {};
        for (const feat of f.features) {
          init[feat] = String(Number(f.ranges[feat].median.toFixed(3)));
        }
        setValues(init);
      })
      .catch((e) =>
        setLoadErr(
          `Could not reach API at ${API_BASE}. Make sure the backend is running. (${e})`
        )
      );
  }, []);

  const sections = useMemo(
    () => (features ? groupFeatures(features.features) : []),
    [features]
  );

  const orderedModels = useMemo(
    () =>
      metrics ? [...metrics.models].sort((a, b) => b.auc_roc - a.auc_roc) : [],
    [metrics]
  );

  async function handlePredict() {
    if (!features) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const payload: Record<string, number> = {};
      for (const f of features.features) {
        const v = parseFloat(values[f]);
        if (Number.isNaN(v)) {
          throw new Error(`"${f}" must be a number`);
        }
        payload[f] = v;
      }
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: payload }),
      });
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        throw new Error(j.detail || `HTTP ${res.status}`);
      }
      const data: PredictionResp = await res.json();
      setResult(data);
    } catch (e: any) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  function resetToMedian() {
    if (!features) return;
    const init: Record<string, string> = {};
    for (const f of features.features) {
      init[f] = String(Number(features.ranges[f].median.toFixed(3)));
    }
    setValues(init);
    setResult(null);
  }

  function randomPatient() {
    if (!features) return;
    const init: Record<string, string> = {};
    for (const f of features.features) {
      const r = features.ranges[f];
      const sample = r.mean + r.std * (Math.random() * 2 - 1);
      const clamped = Math.max(r.min, Math.min(r.max, sample));
      // Snap categorical/integer features (e.g. sex 0/1) to a discrete value
      // instead of emitting decimals like 0.77.
      const isIntegerFeature =
        Number.isInteger(r.min) &&
        Number.isInteger(r.max) &&
        r.max - r.min <= 10;
      init[f] = isIntegerFeature
        ? String(Math.round(clamped))
        : String(Number(clamped.toFixed(3)));
    }
    setValues(init);
    setResult(null);
  }

  return (
    <div className="container">
      {/* Top app bar */}
      <div className="appbar">
        <div className="logo" aria-hidden>
          {/* Heart pulse icon */}
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
            <path
              d="M3 12h4l2-5 4 10 2-5h6"
              stroke="white"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>
        <div>
          <div className="title">AMI Risk Predictor</div>
          <div className="subtitle">
            Clinical decision support · Acute Myocardial Infarction
          </div>
        </div>
        <div className="spacer" />
        <div className="disclaimer">RESEARCH USE ONLY · NOT A MEDICAL DEVICE</div>
      </div>

      {/* Hero */}
      <header className="hero">
        <h1>
          Predict <span className="accent">AMI risk</span> from a blood panel
        </h1>
        <div className="underline" />
        <p>
          Trained on 18 baseline + cutting-edge models in{" "}
          <code>final.py</code>; the top {metrics?.top_k_served ?? 5} by AUC-ROC
          are served live. Enter a patient&apos;s lab values and get a calibrated
          probability of acute myocardial infarction with per-model agreement.
        </p>
        {metrics && (
          <div style={{ marginTop: 18 }}>
            <span className="badge win">Winner: {metrics.winner}</span>
            <span className="badge">
              AUC{" "}
              {orderedModels[0] ? orderedModels[0].auc_roc.toFixed(3) : "-"}
            </span>
            <span className="badge">{metrics.n_features} features</span>
            <span className="badge">
              {metrics.n_train} train / {metrics.n_test} test
            </span>
          </div>
        )}
      </header>

      {loadErr && <div className="error">{loadErr}</div>}

      <div className="grid grid-2" style={{ marginTop: 12 }}>
        {/* ---------- Left: form + result ---------- */}
        <div className="panel">
          <h2>Patient values</h2>
          {!features ? (
            <p style={{ color: "var(--muted)" }}>Loading features…</p>
          ) : (
            <>
              {sections.map((sec, idx) => (
                <details
                  key={sec.name}
                  className="section"
                  open={idx === 0}
                >
                  <summary>
                    {sec.name}
                    <span className="count">{sec.features.length}</span>
                  </summary>
                  <div className="form-grid">
                    {sec.features.map((f) => (
                      <div className="field" key={f}>
                        <label title={f}>{f}</label>
                        <input
                          type="number"
                          step="any"
                          value={values[f] ?? ""}
                          onChange={(e) =>
                            setValues((v) => ({ ...v, [f]: e.target.value }))
                          }
                        />
                      </div>
                    ))}
                  </div>
                </details>
              ))}

              <div className="row-btns">
                <button onClick={handlePredict} disabled={loading}>
                  {loading ? "Predicting…" : "Predict AMI risk"}
                </button>
                <button className="secondary" onClick={resetToMedian}>
                  Reset (median)
                </button>
                <button className="secondary" onClick={randomPatient}>
                  Random patient
                </button>
              </div>
              {error && <div className="error">{error}</div>}

              {result && (
                <div
                  className={`result-card ${
                    result.prediction === 1 ? "ami" : "ok"
                  }`}
                >
                  <h3>Prediction</h3>
                  <div className="result-head">
                    <ConfidenceRing
                      value={result.probability}
                      positive={result.prediction === 1}
                    />
                    <div className="verdict">
                      <div className="big">
                        {result.prediction === 1 ? "AMI likely" : "No AMI"}
                      </div>
                      <div className="meta">
                        Decision by{" "}
                        <b style={{ color: "var(--ink)" }}>{result.winner}</b>
                      </div>
                      <div className="meta" style={{ marginTop: 4 }}>
                        {result.prediction === 1
                          ? "Recommend further cardiac workup."
                          : "AMI markers within expected range."}
                      </div>
                    </div>
                  </div>

                  <div style={{ marginTop: 22 }}>
                    <h3>Per-model agreement</h3>
                    {result.all_models
                      .slice()
                      .sort((a, b) => b.probability - a.probability)
                      .map((m) => (
                        <div
                          key={m.model}
                          className={`model-row ${
                            m.model === result.winner ? "winner" : ""
                          }`}
                        >
                          <div className="name">{m.model}</div>
                          <div className="bar">
                            <span
                              style={{
                                width: `${m.probability * 100}%`,
                              }}
                            />
                          </div>
                          <div className="pct">
                            {(m.probability * 100).toFixed(1)}%
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* ---------- Right: leaderboard ---------- */}
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <div className="panel">
            <h2>Model leaderboard</h2>
            {!metrics ? (
              <p style={{ color: "var(--muted)" }}>Loading metrics…</p>
            ) : (
              orderedModels.map((m) => (
                <div
                  key={m.name}
                  className={`model-row ${
                    m.name === metrics.winner ? "winner" : ""
                  }`}
                >
                  <div className="name">{m.name}</div>
                  <div className="bar">
                    <span style={{ width: `${m.auc_roc * 100}%` }} />
                  </div>
                  <div className="pct">{m.auc_roc.toFixed(3)}</div>
                </div>
              ))
            )}
            <p
              style={{
                color: "var(--muted)",
                fontSize: 11,
                marginTop: 12,
                lineHeight: 1.5,
              }}
            >
              Sorted by AUC-ROC on a held-out 20% test split. Top{" "}
              {metrics?.top_k_served ?? 5} of 18 models are served.
            </p>
          </div>

          {metrics && (
            <div className="panel">
              <h2>Test metrics (winner)</h2>
              {(() => {
                const w = metrics.models.find((m) => m.name === metrics.winner);
                if (!w) return null;
                const rows: [string, number][] = [
                  ["Accuracy", w.accuracy],
                  ["Precision", w.precision],
                  ["Sensitivity", w.sensitivity],
                  ["Specificity", w.specificity],
                  ["F1-Score", w.f1],
                  ["AUC-ROC", w.auc_roc],
                ];
                return rows.map(([k, v]) => (
                  <div className="model-row" key={k}>
                    <div className="name">{k}</div>
                    <div className="bar">
                      <span style={{ width: `${v * 100}%` }} />
                    </div>
                    <div className="pct">{v.toFixed(3)}</div>
                  </div>
                ));
              })()}
            </div>
          )}
        </div>
      </div>

      <p className="footer">
        AMI Research · final.py serving bundle ·{" "}
        {new Date().getFullYear()}
        <br />
        For research and educational use only. Not intended for clinical
        diagnosis or treatment decisions.
      </p>
    </div>
  );
}
