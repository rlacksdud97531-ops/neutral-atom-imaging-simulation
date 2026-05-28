"""
Neutral Atom Imaging Simulation — Streamlit UI
모든 실험·카메라 파라미터를 사이드바에서 실시간으로 조절하고
시뮬레이션 결과를 바로 확인할 수 있는 인터랙티브 앱
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from neutral_atom_imaging_simulation import Camera, Experiment, ImageGenerator

# ── 페이지 기본 설정 ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Neutral Atom Imaging Simulation",
    layout="wide",
    page_icon="⚛️",
)

st.title("⚛️ Neutral Atom Imaging Simulation")
st.caption("Tweezer Array 이미징 시뮬레이터 — 사이드바에서 파라미터를 조절하고 Run 버튼을 누르세요")

# ── Session State 초기화 (결과 유지용) ────────────────────────────────────────
for key in ["image", "psf", "truth", "metadata"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ══════════════════════════════════════════════════════════════════════════════
#  사이드바 — 파라미터 패널
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ 파라미터 설정")

    # ── 1. 카메라 종류 ─────────────────────────────────────────────────────────
    camera_type = st.radio("카메라 타입", ["EMCCD", "CMOS"], horizontal=True)

    # ── 2. 공통 카메라 설정 ────────────────────────────────────────────────────
    with st.expander("📷 카메라 (공통)", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            res_x = st.number_input("해상도 X (px)", 64, 1024, 512, step=64)
            res_y = st.number_input("해상도 Y (px)", 64, 1024, 512, step=64)
        with c2:
            binning = st.selectbox("Binning", [1, 2, 4, 8], index=1)
            exposure_time = st.number_input(
                "노출 시간 (s)", 0.001, 10.0, 0.08, step=0.001, format="%.3f"
            )

        quantum_efficiency = st.slider(
            "양자 효율 (QE)", 0.0, 1.0, 0.86, 0.01,
            help="카메라 파장 대역에서의 양자 효율 [0, 1]"
        )
        numerical_aperture = st.slider(
            "개구수 (NA)", 0.1, 1.0, 0.65, 0.01,
            help="광학계의 수치 개구수"
        )

        c3, c4 = st.columns(2)
        with c3:
            physical_pixel_size = st.number_input(
                "픽셀 크기 (μm)", 1.0, 50.0, 16.0, step=0.5, format="%.1f",
                help="단일 픽셀의 물리적 크기"
            )
            magnification = st.number_input(
                "배율", 1.0, 500.0, 156.25, step=1.0, format="%.2f"
            )
        with c4:
            bias_clamp = st.number_input("Bias Clamp", 0.0, 1000.0, 200.0, step=1.0)
            preampgain = st.number_input(
                "Preamp Gain", 0.01, 10.0, 0.11, step=0.01, format="%.2f"
            )
        readout_stdev = st.number_input(
            "Readout Std Dev", 0.0, 50.0, 4.0, step=0.5, format="%.1f",
            help="최종 읽기 노이즈의 표준편차"
        )

    # ── 3. 카메라 종류별 설정 ──────────────────────────────────────────────────
    if camera_type == "EMCCD":
        with st.expander("📷 EMCCD 전용 설정", expanded=True):
            dark_current_rate = st.number_input(
                "암전류율 (ph/s)", 0.0, 0.01, 0.00029, step=0.00001, format="%.5f"
            )
            cic_chance = st.number_input(
                "CIC Chance", 0.0, 0.01, 0.00037, step=0.00001, format="%.5f",
                help="픽셀당 Clock-Induced Charge 발생 확률"
            )
            scic_chance = st.number_input(
                "Serial CIC Chance", 0.0, 0.001, 0.00002, step=0.000001, format="%.6f",
                help="이득 레지스터당 직렬 CIC 발생 확률"
            )
            number_gain_reg = st.number_input(
                "이득 레지스터 수", 10, 2000, 536, step=1
            )
            p0 = st.number_input(
                "p0 (2차 전자 확률)", 0.0, 0.5, 0.0106982061,
                step=0.0001, format="%.7f",
                help="(1+p0)^N_reg = EM 이득"
            )
    else:
        with st.expander("📷 CMOS 전용 설정", expanded=True):
            st.caption("⚠️ CMOS 모드는 실험적 지원입니다.")
            dc_alpha = st.number_input(
                "암전류 α (Gamma)", 0.001, 1.0, 0.016, step=0.001, format="%.3f"
            )
            dc_beta = st.number_input(
                "암전류 β (Gamma)", 0.01, 10.0, 1.0, step=0.01, format="%.2f"
            )
            bias_stdev = st.number_input(
                "Bias Std Dev", 0.0, 5.0, 0.1, step=0.01, format="%.2f"
            )
            row_noise_stdev = st.number_input(
                "Row Noise Std Dev", 0.0, 5.0, 0.02, step=0.01, format="%.2f"
            )
            col_noise_scale = st.number_input(
                "Column Noise Scale", 0.0, 5.0, 0.02, step=0.01, format="%.2f"
            )
            flicker_noise_scale = st.number_input(
                "Flicker Noise Scale", 0.0, 5.0, 0.01, step=0.01, format="%.2f"
            )

    # ── 4. Zernike 수차 계수 ───────────────────────────────────────────────────
    with st.expander("🔬 Zernike 수차 계수 (15개)", expanded=False):
        st.caption("Noll 인덱싱 기준 15개 계수 (실제 측정값이 기본값으로 설정됨)")
        zernike_defaults = [
            0.0, 0.0, 0.0,
            0.07232454, 0.00087644, -0.01069755,
            0.00280808, 0.00723265, 0.00436401, 0.00117688,
            0.02449155, -0.00427388, -0.00250116, -0.00477205, -0.00054310,
        ]
        zernike_labels = [
            "Z1: Piston",
            "Z2: Y-Tilt",
            "Z3: X-Tilt",
            "Z4: Defocus",
            "Z5: Oblique Astig.",
            "Z6: Vertical Astig.",
            "Z7: Vertical Coma",
            "Z8: Horizontal Coma",
            "Z9: Vertical Trefoil",
            "Z10: Oblique Trefoil",
            "Z11: Primary Spherical",
            "Z12: Vert. 2nd Astig.",
            "Z13: Obl. 2nd Astig.",
            "Z14: Vert. Quadrafoil",
            "Z15: Obl. Quadrafoil",
        ]
        if st.button("🔄 기본값으로 초기화", key="reset_zernike"):
            for i in range(15):
                st.session_state[f"z{i}"] = float(zernike_defaults[i])

        zernike_coeffs = []
        for i, (lbl, dv) in enumerate(zip(zernike_labels, zernike_defaults)):
            val = st.number_input(
                lbl, -2.0, 2.0, float(dv),
                step=0.001, format="%.5f", key=f"z{i}"
            )
            zernike_coeffs.append(val)

    # ── 5. 실험 설정 ───────────────────────────────────────────────────────────
    with st.expander("⚗️ 실험 설정", expanded=True):
        scattering_rate = st.number_input(
            "산란율 Scattering Rate (ph/s)", 100, 1_000_000, 28000, step=100,
            help="원자 1개가 초당 방출하는 광자 수"
        )
        imaging_wavelength = st.number_input(
            "이미징 파장 (μm)", 0.3, 1.5, 0.4619, step=0.001, format="%.4f"
        )
        stray_light_rate = st.number_input(
            "미광율 Stray Light Rate (ph/s)", 0.0, 1000.0, 0.2, step=0.1, format="%.1f"
        )
        survival_probability = st.slider(
            "생존 확률 Survival Prob.", 0.0, 1.0, 1.0, 0.01,
            help="이미징 중 원자가 살아남을 확률"
        )
        fill_rate = st.slider(
            "충전율 Fill Rate", 0.0, 1.0, 0.5, 0.01,
            help="각 트위저 사이트가 채워질 확률"
        )
        light_source_stdev = st.number_input(
            "광원 Std Dev (px)", 0.0, 20.0, 3.0, step=0.5, format="%.1f",
            help="광원 PSF의 가우시안 폭 (픽셀 단위)"
        )

    # ── 6. 트위저 배열 레이아웃 ────────────────────────────────────────────────
    with st.expander("🔲 트위저 배열 레이아웃", expanded=True):
        coord_mode = st.radio(
            "좌표계",
            ["Camera Space (0–1 정규화)", "Physical Space (μm)"],
            key="coord_mode"
        )
        is_cam = "Camera" in coord_mode

        # 좌표계가 바뀌면 위젯을 새로 생성하기 위해 key에 is_cam 포함
        k = str(is_cam)
        if is_cam:
            sp_x_def, sp_y_def = round(20/512, 4), round(20/512, 4)
            off_x_def, off_y_def = round(50/512, 4), round(55/512, 4)
            sp_max = 1.0
            sp_fmt = "%.4f"
        else:
            sp_x_def, sp_y_def = 5.0, 5.0
            off_x_def, off_y_def = 2.0, 2.0
            sp_max = 500.0
            sp_fmt = "%.2f"

        c5, c6 = st.columns(2)
        with c5:
            sp_x = st.number_input(
                "간격 X (Spacing X)", 0.001, sp_max, sp_x_def,
                step=sp_x_def, format=sp_fmt, key=f"spx_{k}"
            )
            cnt_x = st.number_input("열 수 (Columns)", 1, 100, 20, step=1, key=f"cntx_{k}")
            off_x = st.number_input(
                "오프셋 X (Offset X)", 0.0, sp_max, off_x_def,
                step=sp_x_def, format=sp_fmt, key=f"offx_{k}"
            )
            ang_x = st.number_input(
                "각도 X (rad)", -3.14159, 3.14159, 0.0,
                step=0.01, format="%.4f", key=f"angx_{k}"
            )
        with c6:
            sp_y = st.number_input(
                "간격 Y (Spacing Y)", 0.001, sp_max, sp_y_def,
                step=sp_y_def, format=sp_fmt, key=f"spy_{k}"
            )
            cnt_y = st.number_input("행 수 (Rows)", 1, 100, 19, step=1, key=f"cnty_{k}")
            off_y = st.number_input(
                "오프셋 Y (Offset Y)", 0.0, sp_max, off_y_def,
                step=sp_y_def, format=sp_fmt, key=f"offy_{k}"
            )
            ang_y = st.number_input(
                "각도 Y (rad)", -3.14159, 3.14159, 0.0,
                step=0.01, format="%.4f", key=f"angy_{k}"
            )

    # ── 7. 시뮬레이션 정밀도 ───────────────────────────────────────────────────
    approx_steps = st.slider(
        "Approximation Steps", 1, 10, 1,
        help="픽셀 세분화 단계 수. 높을수록 정확하지만 느림"
    )

    st.divider()
    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  시뮬레이션 실행
# ══════════════════════════════════════════════════════════════════════════════
if run_btn:
    with st.spinner("시뮬레이션 실행 중..."):
        try:
            gen = ImageGenerator.ImageGenerator()

            # 카메라 설정
            zernike_arr = np.array(zernike_coeffs, dtype=np.float64)

            if camera_type == "EMCCD":
                cam = Camera.EMCCDCamera(
                    resolution=(int(res_x), int(res_y)),
                    dark_current_rate=dark_current_rate,
                    cic_chance=cic_chance,
                    quantum_efficiency=quantum_efficiency,
                    numerical_aperture=numerical_aperture,
                    physical_pixel_size=physical_pixel_size,
                    magnification=magnification,
                    bias_clamp=bias_clamp,
                    preampgain=preampgain,
                    scic_chance=scic_chance,
                    readout_stdev=readout_stdev,
                    number_gain_reg=int(number_gain_reg),
                    p0=p0,
                    exposure_time=exposure_time,
                    binning=binning,
                )
            else:
                cam = Camera.CMOSCamera(
                    resolution=(int(res_x), int(res_y)),
                    dark_current_sampling_alpha=dc_alpha,
                    dark_current_sampling_beta=dc_beta,
                    quantum_efficiency=quantum_efficiency,
                    numerical_aperture=numerical_aperture,
                    physical_pixel_size=physical_pixel_size,
                    magnification=magnification,
                    bias_clamp=bias_clamp,
                    bias_stdev=bias_stdev,
                    row_noise_stdev=row_noise_stdev,
                    column_noise_scale=col_noise_scale,
                    flicker_noise_scale=flicker_noise_scale,
                    preampgain=preampgain,
                    readout_stdev=readout_stdev,
                    exposure_time=exposure_time,
                    binning=binning,
                )

            cam.set_zernike_coefficients(zernike_arr)
            gen.set_camera(cam)

            # 실험 설정
            exp = Experiment.TweezerArray(
                stray_light_rate=stray_light_rate,
                imaging_wavelength=imaging_wavelength,
                scattering_rate=scattering_rate,
                survival_probability=survival_probability,
                fill_rate=fill_rate,
                light_source_stdev=light_source_stdev,
            )
            if is_cam:
                exp.configure_atom_sites_camera_space(
                    (sp_x, sp_y),
                    (int(cnt_x), int(cnt_y)),
                    (off_x, off_y),
                    (ang_x, ang_y),
                )
            else:
                exp.configure_atom_sites_physical_space(
                    (sp_x, sp_y),
                    (int(cnt_x), int(cnt_y)),
                    (off_x, off_y),
                    (ang_x, ang_y),
                )
            gen.set_experiment(exp)

            # 이미지 생성
            image, truth = gen.create_image(approximation_steps=int(approx_steps))
            psf = gen.get_psf(50)

            # 결과 저장
            st.session_state.image = image
            st.session_state.psf = psf
            st.session_state.truth = truth
            st.session_state.metadata = {
                "camera_type": camera_type,
                "wavelength": imaging_wavelength,
                "na": numerical_aperture,
                "exposure": exposure_time,
                "binning": binning,
            }
            st.success("✅ 시뮬레이션 완료!")

        except Exception as e:
            st.error(f"❌ 시뮬레이션 오류: {e}")
            st.exception(e)

# ══════════════════════════════════════════════════════════════════════════════
#  결과 표시 (세션에 저장된 결과를 항상 표시)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.image is not None:
    image = st.session_state.image
    psf = st.session_state.psf
    truth = st.session_state.truth
    meta = st.session_state.metadata

    atoms_present = int(truth.sum())
    total_sites = len(truth)

    # ── 요약 지표 ───────────────────────────────────────────────────────────
    st.divider()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("카메라", meta["camera_type"])
    m2.metric("원자 수", f"{atoms_present} / {total_sites}")
    m3.metric("실제 충전율", f"{atoms_present / total_sites * 100:.1f}%")
    m4.metric("최대 카운트", f"{image.max():,}")
    m5.metric("이미지 크기", f"{image.shape[0]} × {image.shape[1]}")

    # ── 메인 이미지 + PSF ────────────────────────────────────────────────────
    col_img, col_psf = st.columns([3, 1])

    with col_img:
        st.subheader(f"시뮬레이션 이미지 ({meta['camera_type']})")
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(image, cmap="inferno", origin="upper", aspect="equal")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Counts", rotation=270, labelpad=15)
        ax.set_xlabel("X (pixel)")
        ax.set_ylabel("Y (pixel)")
        ax.set_title(
            f"λ = {meta['wavelength']:.4f} μm  |  NA = {meta['na']:.2f}  |  "
            f"t = {meta['exposure']:.3f} s  |  Binning = {meta['binning']}"
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_psf:
        st.subheader("PSF")
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        im2 = ax2.imshow(psf, cmap="hot")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_title("50 × 50 px")
        ax2.axis("off")
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        st.subheader("사이트 점유율")
        occ_pct = atoms_present / total_sites
        st.progress(occ_pct)
        st.caption(f"{atoms_present} / {total_sites} 사이트 채워짐 ({occ_pct*100:.1f}%)")

    # ── 이미지 강도 히스토그램 ────────────────────────────────────────────────
    with st.expander("📊 픽셀 강도 히스토그램"):
        fig3, ax3 = plt.subplots(figsize=(8, 3))
        flat = image.flatten()
        ax3.hist(flat, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
        ax3.set_xlabel("Pixel Counts")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Pixel Intensity Distribution")
        ax3.axvline(flat.mean(), color="red", linestyle="--", label=f"Mean = {flat.mean():.1f}")
        ax3.legend()
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

    # ── 1D 단면 (Cross-section) ───────────────────────────────────────────────
    with st.expander("📈 수평 / 수직 단면"):
        cx = image.shape[1] // 2
        cy = image.shape[0] // 2
        fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 3))

        ax4a.plot(image[cy, :], color="royalblue")
        ax4a.set_title(f"수평 단면 (row = {cy})")
        ax4a.set_xlabel("X (pixel)")
        ax4a.set_ylabel("Counts")

        ax4b.plot(image[:, cx], color="tomato")
        ax4b.set_title(f"수직 단면 (col = {cx})")
        ax4b.set_xlabel("Y (pixel)")
        ax4b.set_ylabel("Counts")

        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)

else:
    st.info("👈 왼쪽 사이드바에서 파라미터를 설정하고 **▶ Run Simulation** 버튼을 누르세요.")
