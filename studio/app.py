import os
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Micro ML Lab Studio", page_icon="🧪", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono&family=Manrope:wght@400;500;600;700;800&display=swap');
.stApp{background:radial-gradient(ellipse at 75% -10%,#29234b 0%,#111827 45%,#0b1020 100%);font-family:Manrope,sans-serif;color:#edf0ff}
[data-testid="stSidebar"]{background:#101626;border-right:1px solid #29334d}
[data-testid="stMetric"]{background:linear-gradient(145deg,#1b2438,#151b2b);border:1px solid #303a55;padding:16px;border-radius:14px}
[data-testid="stMetricValue"]{font-family:'DM Mono',monospace;color:#f4f3ff}
.stButton>button{border-radius:10px;border:1px solid #796be0;background:linear-gradient(110deg,#7969e8,#9a78e8);color:white;font-weight:700;min-height:42px}
.stTextInput input,.stTextArea textarea{background:#11192a;color:#edf0ff;border:1px solid #303b57;border-radius:10px}
.hero{padding:26px 30px;border:1px solid #393354;border-radius:20px;background:linear-gradient(115deg,rgba(119,99,220,.23),rgba(28,38,61,.8));margin-bottom:22px}
.hero h1{font-size:34px;letter-spacing:-1.4px;margin:0;color:#f4f2ff}.hero p{color:#aab3cb;margin:7px 0 0}
.eyebrow{color:#ad9fff;font:11px 'DM Mono';letter-spacing:2px;text-transform:uppercase;margin-bottom:8px}
</style>
""", unsafe_allow_html=True)


def hero(tag, title, subtitle):
    st.markdown(f'<div class="hero"><div class="eyebrow">{tag}</div><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def api(method, path, **kwargs):
    base = st.session_state.api_url.rstrip("/")
    timeout = kwargs.pop("timeout", 600)
    try:
        response = requests.request(method, base + path, timeout=timeout, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        raise RuntimeError(f"{path}: {detail}") from exc


if "api_url" not in st.session_state:
    st.session_state.api_url = os.getenv("MICRO_ML_API_URL", "http://127.0.0.1:8000")

with st.sidebar:
    st.markdown("## 🧪 Micro ML Lab")
    st.caption("LOCAL INFERENCE WORKBENCH")
    st.session_state.api_url = st.text_input("Backend API", st.session_state.api_url)
    page = st.radio("WORKSPACE", ["Overview", "Model Studio", "Benchmark Lab", "Experiments", "System"])
    st.divider()
    try:
        runtime_status = api("GET", "/runtime/status", timeout=5)
        active = runtime_status.get("active_model")
        if active:
            st.success("Модель загружена")
            st.caption(active["model_id"])
            if st.button("Выгрузить модель", use_container_width=True):
                api("POST", "/runtime/unload")
                st.rerun()
        else:
            st.caption("Runtime готов · модель не загружена")
    except RuntimeError:
        st.error("Backend недоступен")

if page == "Overview":
    hero("YOUR PERSONAL ML WORKSPACE", "Лаборатория начинается здесь.", "Управляй моделями через единый локальный inference runtime.")
    try:
        status = api("GET", "/runtime/status")
        resources = status["resources"]
        gpus = resources.get("gpus", [])
        history = api("GET", "/runtime/benchmarks", params={"limit": 100})
        a, b, c, d = st.columns(4)
        a.metric("Runtime", status["state"].upper(), "CUDA" if status["cuda_available"] else "CPU")
        b.metric("Свободная RAM", f'{resources["available_ram_gb"]:.1f} GB', f'из {resources["total_ram_gb"]:.1f} GB')
        c.metric("GPU", gpus[0]["name"] if gpus else "Не обнаружена")
        d.metric("Benchmarks", len(history), "локальная история")
        if status.get("active_model"):
            st.markdown("### Активная модель")
            st.json(status["active_model"])
        x, y, z = st.columns(3)
        x.info("**Model Studio**\n\nЗагрузка LLM/BERT и выполнение inference.")
        y.info("**Benchmark Lab**\n\nLatency, throughput и VRAM.")
        z.info("**Experiments**\n\nИстория измерений и экспорт.")
    except Exception as exc:
        st.error(str(exc))

elif page == "Model Studio":
    hero("MODEL PLAYGROUND", "Model Studio", "Загружай локальные checkpoints или модели Hugging Face.")
    with st.container(border=True):
        model_id = st.text_input("Локальный путь или Hugging Face ID", placeholder="D:/ml_models/Qwen2.5-0.5B-Instruct")
        a, b, c = st.columns(3)
        task_label = a.selectbox("Тип модели", ["LLM · генерация", "BERT · masked LM", "BERT · классификация", "Encoder · embeddings"])
        task = {"LLM · генерация": "causal", "BERT · masked LM": "masked_lm", "BERT · классификация": "classification", "Encoder · embeddings": "embeddings"}[task_label]
        device = b.selectbox("Устройство", ["auto", "cuda", "cpu"])
        dtype = c.selectbox("Точность", ["auto", "float16", "bfloat16", "float32"], index=1)
        local_only = st.checkbox("Только локальные файлы (без скачивания)", value=True)
        if st.button("Загрузить в runtime", type="primary", use_container_width=True):
            try:
                with st.spinner("Загрузка модели. Это может занять несколько минут..."):
                    info = api("POST", "/runtime/load", json={"model_id": model_id, "task": task, "device": device, "dtype": dtype, "local_files_only": local_only}, timeout=1800)
                st.success("Модель загружена")
                st.json(info)
            except Exception as exc:
                st.error(str(exc))
    try:
        info = api("GET", "/runtime/model", timeout=5)
        st.markdown("### Активная модель")
        st.json(info)
        prompt = st.text_area("Входной текст", "Расскажи кратко, что такое машинное обучение.", height=120)
        max_tokens = st.slider("Максимум новых токенов", 16, 512, 128, step=16)
        sampling = st.checkbox("Использовать sampling", value=False)
        if st.button("▶ Выполнить inference", type="primary"):
            with st.spinner("Модель выполняет задачу..."):
                result = api("POST", "/runtime/infer", json={"text": prompt, "max_new_tokens": max_tokens, "do_sample": sampling})
            st.markdown("#### Результат")
            if isinstance(result["output"], str):
                st.code(result["output"])
            else:
                st.json(result["output"])
            a, b, c = st.columns(3)
            a.metric("Latency", f'{result["latency_seconds"] * 1000:.1f} ms')
            b.metric("Входные токены", result["input_tokens"])
            c.metric("Throughput", f'{result["tokens_per_second"]:.2f} tok/s')
    except RuntimeError:
        st.info("Загрузи модель, чтобы открыть playground.")
    except Exception as exc:
        st.error(str(exc))

elif page == "Benchmark Lab":
    hero("PERFORMANCE ENGINEERING", "Benchmark Lab", "Прогрев, повторные измерения и история на стороне backend.")
    try:
        info = api("GET", "/runtime/model", timeout=5)
        st.caption(f'{info["model_id"]} · {info["task"]} · {info["device"]}')
        prompt = st.text_area("Тестовый prompt", "Explain what a transformer neural network is in simple terms.")
        a, b, c = st.columns(3)
        iterations = a.slider("Измерений", 1, 30, 5)
        warmups = b.slider("Прогрев", 0, 5, 1)
        tokens = c.slider("Новых токенов", 16, 256, 64, step=16)
        if st.button("⚡ Запустить benchmark", type="primary", use_container_width=True):
            with st.spinner("Выполняется benchmark..."):
                result = api("POST", "/runtime/benchmark", json={"prompt": prompt, "iterations": iterations, "warmup_runs": warmups, "max_new_tokens": tokens}, timeout=1800)
            st.success("Результат сохранён движком")
            x, y, z, w = st.columns(4)
            x.metric("Средняя latency", f'{result["mean_latency_ms"]:.1f} ms')
            y.metric("Медиана", f'{result["median_latency_ms"]:.1f} ms')
            z.metric("Throughput", f'{result["throughput_tokens_per_second"]:.2f} tok/s')
            w.metric("Peak VRAM", f'{result["peak_cuda_allocated_bytes"] / 1024**3:.2f} GB')
            st.line_chart(pd.DataFrame({"Latency, ms": [v * 1000 for v in result["samples_seconds"]]}))
            st.json(result)
    except RuntimeError:
        st.warning("Сначала загрузи модель в Model Studio.")
    except Exception as exc:
        st.error(str(exc))

elif page == "Experiments":
    hero("RESEARCH LOGBOOK", "Experiments", "История benchmark хранится на стороне локального backend.")
    try:
        history = api("GET", "/runtime/benchmarks", params={"limit": 500})
        if history:
            df = pd.DataFrame(history)
            a, b = st.columns(2)
            a.metric("Всего запусков", len(df))
            b.metric("Средняя latency", f'{df["mean_latency_ms"].mean():.1f} ms')
            st.dataframe(df.iloc[::-1], use_container_width=True, hide_index=True)
            if "throughput_tokens_per_second" in df:
                chart = df.reset_index().rename(columns={"index": "Run", "throughput_tokens_per_second": "tok/s"}).set_index("Run")[["tok/s"]]
                st.bar_chart(chart)
            st.download_button("Скачать JSON", df.to_json(orient="records", force_ascii=False, indent=2), "micro-ml-benchmarks.json", "application/json")
        else:
            st.info("Экспериментов пока нет.")
    except Exception as exc:
        st.error(str(exc))

elif page == "System":
    hero("HARDWARE TELEMETRY", "System Monitor", "Ресурсы и runtime status, полученные от backend.")
    try:
        status = api("GET", "/runtime/status")
        resources = status["resources"]
        a, b, c = st.columns(3)
        a.metric("CPU threads", resources["cpu_count"])
        b.metric("RAM total", f'{resources["total_ram_gb"]:.2f} GB')
        c.metric("RAM available", f'{resources["available_ram_gb"]:.2f} GB')
        for gpu in resources.get("gpus", []):
            st.markdown(f'### {gpu["name"]}')
            x, y = st.columns(2)
            x.metric("VRAM total", f'{gpu["total_vram_gb"]:.2f} GB')
            y.metric("VRAM free", f'{gpu["free_vram_gb"]:.2f} GB')
            st.progress(min(1 - gpu["free_vram_gb"] / max(gpu["total_vram_gb"], 0.01), 1.0))
        st.json({"runtime": status["state"], "PyTorch": status["torch"], "CUDA": status["cuda_version"], "CUDA available": status["cuda_available"]})
    except Exception as exc:
        st.error(str(exc))
