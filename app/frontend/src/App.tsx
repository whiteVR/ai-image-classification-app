import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type Task = "mnist" | "cats-dogs";

type Probability = {
  label: string;
  probability: number;
};

type PredictionResult = {
  task: string;
  variant?: string;
  variantLabel?: string;
  prediction: string;
  confidence: number;
  probabilities: Probability[];
  modelPath: string;
};

type ModelStatus = {
  mnist: {
    path: string;
    exists: boolean;
  };
  catsDogs: Record<
    string,
    {
      label: string;
      path: string;
      exists: boolean;
    }
  >;
};

const API_BASE = "";

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

function App() {
  const [task, setTask] = useState<Task>("mnist");
  const [variant, setVariant] = useState("finetuned");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const endpoint = useMemo(() => {
    if (task === "mnist") {
      return `${API_BASE}/api/predict/mnist`;
    }

    return `${API_BASE}/api/predict/cats-dogs?variant=${encodeURIComponent(variant)}`;
  }, [task, variant]);

  useEffect(() => {
    fetch(`${API_BASE}/api/models`)
      .then((response) => response.json())
      .then((data) => setModelStatus(data))
      .catch(() => setModelStatus(null));
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  function handleTaskChange(nextTask: Task) {
    setTask(nextTask);
    setResult(null);
    setError(null);
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const selectedFile = event.target.files?.[0] ?? null;
    setFile(selectedFile);
    setResult(null);
    setError(null);

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setPreviewUrl(selectedFile ? URL.createObjectURL(selectedFile) : null);
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!file) {
      setError("예측할 이미지 파일을 선택하세요.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail ?? "예측 요청이 실패했습니다.");
      }

      setResult(data);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "예측 요청이 실패했습니다.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <div className="topbar">
          <div>
            <h1>AI 이미지 분류 실습 앱</h1>
            <p>MNIST 숫자 분류와 Cats/Dogs 분류 모델을 같은 화면에서 실행합니다.</p>
          </div>
          <div className="status-pill">{modelStatus ? "API 연결됨" : "API 확인 중"}</div>
        </div>

        <div className="task-tabs" role="tablist" aria-label="분류 작업 선택">
          <button
            className={task === "mnist" ? "active" : ""}
            type="button"
            onClick={() => handleTaskChange("mnist")}
          >
            MNIST 숫자
          </button>
          <button
            className={task === "cats-dogs" ? "active" : ""}
            type="button"
            onClick={() => handleTaskChange("cats-dogs")}
          >
            Cats / Dogs
          </button>
        </div>

        <div className="content-grid">
          <form className="panel uploader" onSubmit={handleSubmit}>
            <div className="panel-title">
              <h2>이미지 업로드</h2>
              <span>{task === "mnist" ? "28x28 흑백 전처리" : "160x160 RGB 전처리"}</span>
            </div>

            {task === "cats-dogs" && (
              <label className="field">
                <span>모델 선택</span>
                <select value={variant} onChange={(event) => setVariant(event.target.value)}>
                  <option value="finetuned">MobileNetV2 fine-tuned</option>
                  <option value="mobilenetv2">MobileNetV2 feature extraction</option>
                  <option value="cnn">Kaggle CNN baseline</option>
                </select>
              </label>
            )}

            <label className="drop-zone">
              <input type="file" accept="image/*" onChange={handleFileChange} />
              {previewUrl ? (
                <img src={previewUrl} alt="업로드 미리보기" />
              ) : (
                <span>이미지 파일을 선택하세요</span>
              )}
            </label>

            <button className="primary-button" type="submit" disabled={isLoading}>
              {isLoading ? "예측 중..." : "예측 실행"}
            </button>

            {error && <div className="error-box">{error}</div>}
          </form>

          <section className="panel result-panel">
            <div className="panel-title">
              <h2>예측 결과</h2>
              <span>{result?.modelPath ?? "모델 대기 중"}</span>
            </div>

            {result ? (
              <>
                <div className="prediction-card">
                  <span className="label">Prediction</span>
                  <strong>{result.prediction}</strong>
                  <span className="confidence">{formatPercent(result.confidence)}</span>
                </div>

                {result.variantLabel && <p className="variant-label">{result.variantLabel}</p>}

                <div className="bars">
                  {result.probabilities.map((item) => (
                    <div className="bar-row" key={item.label}>
                      <div className="bar-meta">
                        <span>{item.label}</span>
                        <span>{formatPercent(item.probability)}</span>
                      </div>
                      <div className="bar-track">
                        <div
                          className="bar-fill"
                          style={{ width: `${Math.max(item.probability * 100, 1)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="empty-result">왼쪽에서 이미지를 업로드하고 예측을 실행하세요.</div>
            )}
          </section>
        </div>

        {modelStatus && (
          <section className="model-table">
            <h2>모델 파일 상태</h2>
            <div className="model-row">
              <span>MNIST CNN</span>
              <code>{modelStatus.mnist.path}</code>
              <strong>{modelStatus.mnist.exists ? "있음" : "없음"}</strong>
            </div>
            {Object.entries(modelStatus.catsDogs).map(([key, value]) => (
              <div className="model-row" key={key}>
                <span>{value.label}</span>
                <code>{value.path}</code>
                <strong>{value.exists ? "있음" : "없음"}</strong>
              </div>
            ))}
          </section>
        )}
      </section>
    </main>
  );
}

createRoot(document.getElementById("root") as HTMLElement).render(<App />);
