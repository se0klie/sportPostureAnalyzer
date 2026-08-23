
import os
import tempfile
from pathlib import Path

import streamlit as st
from main import process_video


st.set_page_config(
    page_title="Analizador de posturas deportivas",
    page_icon="🏋️",
    layout="wide",
)

st.title("🏋️ Analizador de posturas deportivas")
st.write(
    "Sube un video, selecciona el ejercicio a realizar y el sistema lo procesará"
    "con la detección de pose, errores de postura, fases de descanso/neutral y recomendaciones."
)

with st.sidebar:
    st.header("Configuración")

    exercise = st.selectbox(
        "Selecciona el ejercicio",
        ["squat", "push-up", "plank"],
        format_func=lambda x: {
            "squat": "Sentadilla",
            "push-up": "Flexiones",
            "plank": "Plancha",
        }[x],
    )

    st.info(
        "Ejercicios disponibles a analizar: Sentadilla, flexiones y plancha."
    )

uploaded_file = st.file_uploader(
    "Sube tu video",
    type=["mp4", "mov", "avi", "mkv", "webm"],
)

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Procesar video", type="primary", use_container_width=True):
        input_path = None
        output_path = None

        try:
            suffix = Path(uploaded_file.name).suffix or ".mp4"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                input_path = tmp.name

            output_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            )
            output_path = output_file.name
            output_file.close()

            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(progress):
                progress = max(0.0, min(float(progress), 1.0))
                progress_bar.progress(progress)
                status_text.text(
                    f"Procesando tu video... {int(progress * 100)}%"
                )

            with st.spinner("Analizando tu postura..."):
                processed_path, summary = process_video(
                    input_path,
                    exercise,
                    output_path,
                    progress_cb=update_progress,
                )

            progress_bar.progress(1.0)
            status_text.success("Video procesado exitosamente!")

            # Results
            st.divider()
            st.header("📊 Resultados")

            active_frames = summary.get("active_frames", 0)
            correct_frames = summary.get("correct_frames", 0)
            score = summary.get("score_pct")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Frames analizados", active_frames)

            with col2:
                st.metric("Frames correctos", correct_frames)

            with col3:
                st.metric(
                    "Posture score",
                    f"{score:.1f}%" if score is not None else "N/A",
                )

            st.subheader("Video procesado!")
            st.video(processed_path)

            st.subheader("Análisis de tus articulaciones")

            per_joint = summary.get("per_joint", {})

            if per_joint:
                for joint, data in per_joint.items():
                    percentage = data.get("ok_pct")

                    if percentage is None:
                        st.write(f"**{joint.capitalize()}**: No hay error, muy bien! Tu técnica fue impecable")
                    else:
                        st.write(
                            f"**{joint.capitalize()}** — "
                            f"{percentage:.1f}% correcto"
                        )
                        st.progress(int(min(max(percentage, 0), 100)))

            st.subheader("Errores y recomendaciones")

            messages = summary.get("top_messages", [])

            if messages:
                for item in messages:
                    message = item["message"]
                    count = item["count"]

                    st.error(
                        f"{message}  \n"
                        f"Detectado en casi {count} frames activos."
                    )
            else:
                st.success(
                    "No se encontraron errores en las fases activas, muy bien!"
                )

            total_frames = summary.get("total_frames", 0)
            neutral_frames = max(total_frames - active_frames, 0)

            st.subheader("Análisis de la fase de descanso/neutra")

            if neutral_frames > 0:
                st.info(
                    f"{neutral_frames} frames clasificados como inactivos, "
                    "descanso, or preparación en vez de una fase activa de ejercicio."
                )
            else:
                st.info("No hubo data en fase inactiva/neutral.")

            st.subheader("Ángulos detectados!")

            ranges = summary.get("ranges", {})

            if ranges:
                for joint, values in ranges.items():
                    minimum = values.get("min")
                    maximum = values.get("max")

                    if minimum != float("inf") and maximum != float("-inf"):
                        st.write(
                            f"**{joint.capitalize()}**: "
                            f"{minimum:.1f}° – {maximum:.1f}°"
                        )

            # Download
            st.divider()

            with open(processed_path, "rb") as video_file:
                st.download_button(
                    label="⬇️ Descargar vídeo procesado",
                    data=video_file,
                    file_name=f"{exercise}_processed.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"No se pudo procesar el vídeo: {e}")

        finally:
            if input_path and os.path.exists(input_path):
                try:
                    os.unlink(input_path)
                except OSError:
                    pass
