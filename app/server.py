import numpy as np
import pandas as pd
import cv2
import streamlit as st

from corseacare.config import Config
from corseacare.runner import build_pipeline
from corseacare.viz import draw_overlay


def run_on_image(image_bgr: np.ndarray, mm_per_px: float):
    cfg = Config(); cfg.mm_per_px = mm_per_px
    pipe = build_pipeline(cfg, mm_per_px)
    r = pipe.run(image_bgr)
    df = pd.DataFrame(r["records"])
    overlay = draw_overlay(image_bgr, r["detections"], r["masks"])
    return r["count"], df, overlay


def main():
    st.title("CorSeaCare — comptage de particules plastiques")
    mm_per_px = st.number_input("Échelle (mm/pixel)", value=0.1, format="%.4f")
    files = st.file_uploader("Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    all_rows = []
    for f in files or []:
        arr = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        count, df, overlay = run_on_image(arr, mm_per_px)
        st.subheader(f"{f.name} — {count} particules")
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        st.dataframe(df)
        df["image"] = f.name; all_rows.append(df)
    if all_rows:
        full = pd.concat(all_rows, ignore_index=True)
        st.download_button("Exporter CSV", full.to_csv(index=False), "corseacare_counts.csv")


if __name__ == "__main__":
    main()
