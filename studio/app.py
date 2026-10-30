import json
import sys
import streamlit as st
import pandas as pd
from engine import benchmark, benchmark_history, discover_local_models, hardware_info, load_model, run_inference

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

if "model" not in st.session_state:
    st.session_state.update(model=None, tokenizer=None, task=None, model_id=None, device=None)

with st.sidebar:
    st.markdown("## 🧪 Micro ML Lab")
    st.caption("LOCAL AI WORKBENCH · v0.2")
    page = st.radio("WORKSPACE", ["Overview", "Model Studio", "Benchmark Lab", "Experiments", "System"])
    st.divider()
    if st.session_state.model is not None:
        st.success("Модель загружена")
        st.caption(str(st.session_state.model_id))
        if st.button("Выгрузить модель", use_container_width=True):
            import torch
            st.session_state.update(model=None, tokenizer=None, task=None, model_id=None, device=None)
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            st.rerun()
    else:
        st.caption("Нет активной модели")

if page == "Overview":
    hero("YOUR PERSONAL ML WORKSPACE", "Лаборатория начинается здесь.", "Запускай модели локально, измеряй производительность и сохраняй эксперименты.")
    info, history = hardware_info(), benchmark_history()
    a,b,c,d=st.columns(4)
    a.metric("Устройство",info["device"].upper(),"CUDA доступна" if info["cuda"] else "CPU режим")
    b.metric("Свободная RAM",f'{info["ram_available_gb"]:.1f} GB',f'из {info["ram_total_gb"]:.1f} GB')
    gpu=info["gpus"][0] if info["gpus"] else None
    c.metric("GPU",gpu["name"] if gpu else "Не обнаружена")
    d.metric("Эксперименты",len(history),"локальная история")
    st.markdown("### Рабочие модули")
    x,y,z=st.columns(3)
    x.info("**Model Studio**\n\nЗагрузка LLM/BERT, генерация, классификация и embeddings.")
    y.info("**Benchmark Lab**\n\nLatency, tokens/sec, VRAM и повторяемые тесты.")
    z.info("**Experiments**\n\nИстория запусков и экспорт результатов.")
    if history: st.dataframe(pd.DataFrame(history).tail(8).iloc[::-1],use_container_width=True,hide_index=True)
    else: st.caption("Проведи первый benchmark, чтобы здесь появилась история.")

elif page == "Model Studio":
    hero("MODEL PLAYGROUND","Model Studio","Локальные LLM и BERT в одном интерфейсе.")
    models=discover_local_models()
    if models: st.caption(f"Обнаружено локальных папок с config.json: {len(models)}")
    paths=[""]+[m["path"] for m in models]
    chosen=st.selectbox("Локальные модели",paths,format_func=lambda p: next((m["name"] for m in models if m["path"]==p),"Выбрать..."))
    left,right=st.columns([2,1])
    with left: model_id=st.text_input("Путь к модели или Hugging Face ID",value=chosen,placeholder="D:/ml_models/Qwen2.5-0.5B-Instruct")
    with right:
        task_label=st.selectbox("Задача",["LLM · генерация","BERT · masked LM","BERT · классификация","Encoder · embeddings"])
        task={"LLM · генерация":"causal","BERT · masked LM":"masked","BERT · классификация":"classification","Encoder · embeddings":"embeddings"}[task_label]
        dtype=st.selectbox("Точность",["auto","float16","bfloat16","float32"])
        device=st.selectbox("Устройство",["auto","cuda","cpu"])
    if st.button("Загрузить модель",type="primary",use_container_width=True):
        try:
            with st.spinner("Загрузка модели и токенизатора..."):
                model,tokenizer,actual=load_model(model_id,task,dtype,device)
            st.session_state.update(model=model,tokenizer=tokenizer,task=task,model_id=model_id,device=actual)
            st.success(f"Модель загружена на {actual}")
        except Exception as exc: st.error(f"Ошибка загрузки: {exc}")
    if st.session_state.model is not None:
        st.divider()
        st.markdown(f"#### Активная модель: {st.session_state.model_id}")
        default="The capital of France is [MASK]." if st.session_state.task=="masked" else "Расскажи кратко, что такое машинное обучение."
        prompt=st.text_area("Входной текст",default,height=130)
        tokens=st.slider("Новых токенов",16,512,128,step=16) if st.session_state.task=="causal" else 1
        if st.button("▶ Запустить инференс",type="primary"):
            try:
                with st.spinner("Выполняется инференс..."):
                    result=run_inference(st.session_state.model,st.session_state.tokenizer,st.session_state.task,prompt,tokens)
                st.markdown("#### Результат")
                st.code(result["output"])
                a,b,c=st.columns(3)
                a.metric("Latency",f'{result["latency_seconds"]*1000:.1f} ms')
                b.metric("Входные токены",result["input_tokens"])
                c.metric("Скорость",f'{result["tokens_per_second"]:.2f} tok/s')
            except Exception as exc: st.error(str(exc))
    else: st.info("Укажи модель и загрузи её. Поддерживаются локальные папки и Hugging Face ID.")

elif page == "Benchmark Lab":
    hero("PERFORMANCE ENGINEERING","Benchmark Lab","Прогрев, повторные измерения и сравнение производительности.")
    if st.session_state.model is None: st.warning("Сначала загрузи модель в Model Studio.")
    else:
        st.caption(f'Активная модель: {st.session_state.model_id} · {st.session_state.task}')
        prompt=st.text_area("Тестовый prompt","Explain what a transformer neural network is in simple terms.")
        a,b=st.columns(2)
        iterations=a.slider("Количество измерений",3,30,5)
        tokens=b.slider("Новых токенов",16,256,64,step=16)
        if st.button("⚡ Запустить benchmark",type="primary",use_container_width=True):
            try:
                with st.spinner("Прогрев и серия измерений..."):
                    result=benchmark(st.session_state.model,st.session_state.tokenizer,st.session_state.task,prompt,iterations,tokens)
                st.success("Результаты сохранены в results/studio_benchmarks.jsonl")
                a,b,c,d=st.columns(4)
                a.metric("Средняя latency",f'{result["mean_latency_ms"]:.1f} ms')
                b.metric("Медиана",f'{result["median_latency_ms"]:.1f} ms')
                c.metric("Throughput",f'{result["mean_tokens_per_second"]:.2f} tok/s')
                d.metric("Peak VRAM",f'{result["peak_vram_gb"]:.2f} GB')
                st.line_chart(pd.DataFrame({"Latency, ms":[v*1000 for v in result["samples_seconds"]]}))
                st.json(result)
            except Exception as exc: st.error(str(exc))
    history=benchmark_history()
    if history:
        st.markdown("### История измерений")
        st.dataframe(pd.DataFrame(history).iloc[::-1],use_container_width=True,hide_index=True)
        st.download_button("Скачать JSON",json.dumps(history,ensure_ascii=False,indent=2),"micro-ml-benchmarks.json","application/json")

elif page == "Experiments":
    hero("RESEARCH LOGBOOK","Experiments","История запусков и результаты для дальнейшего анализа.")
    history=benchmark_history()
    if history:
        df=pd.DataFrame(history)
        a,b=st.columns(2); a.metric("Всего запусков",len(df)); b.metric("Средняя latency",f'{df["mean_latency_ms"].mean():.1f} ms')
        st.dataframe(df.iloc[::-1],use_container_width=True,hide_index=True)
        st.bar_chart(df.reset_index().rename(columns={"index":"Run","mean_tokens_per_second":"tok/s"}).set_index("Run")[["tok/s"]])
    else: st.info("Экспериментов пока нет. Запусти тест в Benchmark Lab.")

elif page == "System":
    hero("HARDWARE TELEMETRY","System Monitor","Ресурсы машины и состояние CUDA.")
    info=hardware_info()
    a,b,c=st.columns(3)
    a.metric("CPU threads",info["cpu"]); b.metric("RAM total",f'{info["ram_total_gb"]:.2f} GB'); c.metric("RAM available",f'{info["ram_available_gb"]:.2f} GB')
    if info["gpus"]:
        for i,gpu in enumerate(info["gpus"]):
            st.markdown(f'### GPU {i} · {gpu["name"]}')
            a,b,c=st.columns(3)
            a.metric("VRAM total",f'{gpu["vram_total_gb"]:.2f} GB')
            b.metric("Allocated",f'{gpu["vram_allocated_gb"]:.2f} GB')
            c.metric("Reserved",f'{gpu["vram_reserved_gb"]:.2f} GB')
            st.progress(min(gpu["vram_allocated_gb"]/max(gpu["vram_total_gb"],0.01),1.0))
    else: st.warning("CUDA GPU не обнаружена. Доступен CPU режим.")
    import torch
    st.markdown("### Runtime")
    st.json({"Python":sys.version.split()[0],"PyTorch":torch.__version__,"CUDA":torch.version.cuda,"CUDA available":torch.cuda.is_available()})
