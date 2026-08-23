import os
# Protobuf फिक्स - यह MediaPipe को Python मोड में चलाता है
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import json
import base64
import io
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# ---------- MediaPipe को स्टैण्डर्ड तरीके से इम्पोर्ट करें ----------
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ---------- पेज कॉन्फ़िग ----------
st.set_page_config(
    page_title="NPRC B.A.D.R v5.0 - Clinical Rehabilitation",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Web Speech API (Voice Feedback) ----------
def speak_text(text):
    js_code = f"""
    <script>
    function speakNow() {{
        var utterance = new SpeechSynthesisUtterance({text});
        utterance.lang = 'hi-IN';
        utterance.rate = 0.9;
        utterance.pitch = 1;
        window.speechSynthesis.speak(utterance);
    }}
    speakNow();
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)

# ---------- ऑफलाइन ऑथेंटिकेशन ----------
def load_credentials():
    try:
        with open('config.json', 'r') as f:
            data = json.load(f)
            return data.get('username', 'admin'), data.get('password', 'NPRC@2026')
    except:
        return 'admin', 'NPRC@2026'

def login(username, password):
    correct_username, correct_password = load_credentials()
    if username == correct_username and password == correct_password:
        st.session_state.authenticated = True
        st.session_state.user = username
        return True
    return False

def logout():
    st.session_state.authenticated = False
    st.session_state.user = None
    st.rerun()

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user = None

# ---------- CSS थीम ----------
st.markdown("""
<style>
    .stApp { background-color: #F4F7FC; }
    .main-header {
        background: linear-gradient(135deg, #0B2A4A 0%, #1A4B6D 100%);
        padding: 1.2rem 2rem;
        border-radius: 0 0 20px 20px;
        border-bottom: 4px solid #D4AF37;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .badge-clinical {
        background: #D4AF37;
        color: #0B2A4A;
        padding: 2px 16px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.7rem;
    }
    .stat-box {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
        border-bottom: 4px solid #D4AF37;
    }
    .stat-box h2 { color: #0B2A4A; font-size: 2.2rem; font-weight: 800; margin: 0; }
    .stat-box p { color: #4B5563; font-weight: 500; margin: 0; }
    .footer {
        background: #0B2A4A;
        color: #B0C4DE;
        padding: 1.2rem 2rem;
        border-radius: 20px 20px 0 0;
        margin-top: 3rem;
        text-align: center;
        border-top: 4px solid #D4AF37;
    }
    section[data-testid="stSidebar"] { background-color: #0B2A4A !important; }
    section[data-testid="stSidebar"] * { color: white !important; }
    .stButton button {
        background: #D4AF37 !important;
        color: #0B2A4A !important;
        font-weight: 700 !important;
        border-radius: 30px !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

def svg_icon(name, size=24, color="#FFFFFF"):
    icons = {
        "gov": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>',
        "clinical": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M22 12h-4l-3 9-4-18-3 9H2"/></svg>',
        "patient": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
    }
    return icons.get(name, "")

# ---------- हेडर ----------
st.markdown(f"""
<div class="main-header">
    <div style="display:flex; align-items:center; gap:20px;">
        <div style="background:#D4AF37; width:50px; height:50px; border-radius:50%; display:flex; align-items:center; justify-content:center;">
            {svg_icon('gov', 32, '#0B2A4A')}
        </div>
        <div>
            <h1 style="color:white; margin:0; font-weight:300;">NPRC CLINICAL</h1>
            <h2 style="color:white; margin:0; font-weight:700;">B.A.D.R v5.0</h2>
            <p style="color:#B0C4DE; margin:0;">Biomechanical Assessment, Diagnosis & Rehabilitation</p>
        </div>
    </div>
    <div style="text-align:right; color:#D4AF37;">
        <span class="badge-clinical">Govt. Hospital & Sports Med</span>
        <div style="color:#B0C4DE; font-size:0.8rem;">Offline | Secure</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- साइडबार ----------
with st.sidebar:
    st.markdown("---")
    if not st.session_state.authenticated:
        with st.form("login_form"):
            st.text_input("Username", key="u")
            st.text_input("Password", type="password", key="p")
            if st.form_submit_button("Authenticate"):
                if login(st.session_state.u, st.session_state.p):
                    st.success("Access Granted")
                    st.rerun()
                else:
                    st.error("Invalid")
        st.stop()
    else:
        st.markdown(f"✅ Logged in as **{st.session_state.user}**")
        if st.button("Logout", use_container_width=True):
            logout()
        st.markdown("---")
        page = st.radio(
            "Navigation",
            ["Clinical Dashboard", "Assessment", "Patient Records", "Reports", "Settings"]
        )

# ---------- डेटा ----------
if 'patients' not in st.session_state:
    try:
        st.session_state.patients = pd.read_csv('patients.csv').to_dict('records')
    except:
        st.session_state.patients = []

if 'assessments' not in st.session_state:
    try:
        st.session_state.assessments = pd.read_csv('assessments.csv').to_dict('records')
    except:
        st.session_state.assessments = []

def save_patients():
    pd.DataFrame(st.session_state.patients).to_csv('patients.csv', index=False)

def save_assessments():
    pd.DataFrame(st.session_state.assessments).to_csv('assessments.csv', index=False)

# ---------- AI Engine ----------
class KalmanFilter:
    def __init__(self):
        self.k = 0.6
        self.last_val = None
    def update(self, val):
        if self.last_val is None:
            self.last_val = val
            return val
        self.last_val = self.last_val + self.k * (val - self.last_val)
        return self.last_val

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(rad * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

class BADRProcessor(VideoTransformerBase):
    def __init__(self, exercise_type="Squat"):
        self.pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.exercise_type = exercise_type
        self.kf_left = KalmanFilter()
        self.kf_right = KalmanFilter()
        self.scores = {"left": [], "right": []}
        self.last_feedback = ""

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        h, w, _ = img.shape
        feedback = "Stand in front of camera"
        color = (0, 255, 255)
        l_ang = r_ang = 0

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            try:
                l_hip = (lm[mp_pose.PoseLandmark.LEFT_HIP.value].x*w, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y*h)
                l_kn = (lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x*w, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y*h)
                l_ank = (lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x*w, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y*h)
                l_sh = (lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x*w, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y*h)
                l_elbow = (lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x*w, lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y*h)
                l_wrist = (lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x*w, lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y*h)

                r_hip = (lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x*w, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y*h)
                r_kn = (lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x*w, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y*h)
                r_ank = (lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x*w, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y*h)
                r_sh = (lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x*w, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y*h)
                r_elbow = (lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x*w, lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y*h)
                r_wrist = (lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x*w, lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y*h)

                if self.exercise_type == "Squat":
                    l_ang = calculate_angle(l_hip, l_kn, l_ank)
                    r_ang = calculate_angle(r_hip, r_kn, r_ank)
                    l_ang = self.kf_left.update(l_ang)
                    r_ang = self.kf_right.update(r_ang)
                    if l_ang < 160 and r_ang < 160:
                        if 70 < l_ang < 110 and 70 < r_ang < 110:
                            feedback, color = "✅ PERFECT SQUAT - 90°", (0,255,0)
                        elif l_ang < 70 or r_ang < 70:
                            feedback, color = "⚠️ TOO DEEP - Stop at 90°", (0,0,255)
                        else:
                            feedback, color = "🔄 Bend to 90°", (255,255,0)
                    else:
                        feedback, color = "🧍 Stand Straight. Bend knees.", (255,255,0)

                elif self.exercise_type == "Shoulder Flexion":
                    l_ang = calculate_angle(l_sh, l_elbow, l_wrist)
                    r_ang = calculate_angle(r_sh, r_elbow, r_wrist)
                    l_ang = self.kf_left.update(l_ang)
                    r_ang = self.kf_right.update(r_ang)
                    if l_ang < 180 and r_ang < 180:
                        if 150 < l_ang < 180 and 150 < r_ang < 180:
                            feedback, color = "✅ PERFECT - Full Extension", (0,255,0)
                        elif l_ang < 150 or r_ang < 150:
                            feedback, color = "⚠️ Raise arms higher", (0,0,255)
                        else:
                            feedback, color = "🔄 Extend arms fully", (255,255,0)
                    else:
                        feedback, color = "🧍 Arms at side. Raise up.", (255,255,0)

                elif self.exercise_type == "Hip Abduction":
                    l_ang = calculate_angle(l_hip, l_kn, l_ank)
                    r_ang = calculate_angle(r_hip, r_kn, r_ank)
                    l_ang = self.kf_left.update(l_ang)
                    r_ang = self.kf_right.update(r_ang)
                    if l_ang > 170 or r_ang > 170:
                        if l_ang > 170 and r_ang > 170:
                            feedback, color = "✅ PERFECT - Legs Apart", (0,255,0)
                        else:
                            feedback, color = "⚠️ Spread legs wider", (0,0,255)
                    else:
                        feedback, color = "🔄 Step legs apart", (255,255,0)

                elif self.exercise_type == "Balance Test":
                    l_ankle_y = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
                    r_ankle_y = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
                    if abs(l_ankle_y - r_ankle_y) > 0.05:
                        feedback, color = "✅ Good Balance", (0,255,0)
                    else:
                        feedback, color = "⚠️ Lift one leg", (0,0,255)

                self.scores["left"].append(l_ang)
                self.scores["right"].append(r_ang)

            except Exception as e:
                cv2.putText(img, "Error", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            cv2.putText(img, "No body detected", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

        cv2.putText(img, f"L: {int(l_ang)}° | R: {int(r_ang)}°", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(img, feedback, (20,90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        cv2.putText(img, f"Test: {self.exercise_type}", (20,140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        self._feedback_text = feedback
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------- पेज ----------
if page == "Clinical Dashboard":
    st.markdown("## Clinical Dashboard")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='stat-box'><h2>{len(st.session_state.patients)}</h2><p>Total Patients</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='stat-box'><h2>{len(st.session_state.assessments)}</h2><p>Assessments Done</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='stat-box'><h2>B.A.D.R v5.0</h2><p>Clinical Engine</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Quick Assessment")
    exercise = st.selectbox("Select Test", ["Squat", "Shoulder Flexion", "Hip Abduction", "Balance Test"])
    patient_id = st.selectbox("Select Patient", [p['name'] for p in st.session_state.patients] if st.session_state.patients else ["No Patient"])
    if st.button("Start Assessment"):
        st.session_state.exercise = exercise
        st.session_state.current_patient = patient_id
        st.rerun()

elif page == "Assessment":
    st.markdown("## Live Clinical Assessment")
    st.markdown(f"**Patient:** {st.session_state.get('current_patient', 'Not Selected')}")
    st.markdown(f"**Test:** {st.session_state.get('exercise', 'Squat')}")

    RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    ctx = webrtc_streamer(
        key="badr-clinical",
        video_processor_factory=lambda: BADRProcessor(st.session_state.get('exercise', 'Squat')),
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
    )
    if ctx and ctx.video_processor:
        if hasattr(ctx.video_processor, '_feedback_text'):
            current_feedback = ctx.video_processor._feedback_text
            if current_feedback and "Stand" not in current_feedback:
                speak_text(current_feedback)
        if st.button("Save Assessment"):
            scores = ctx.video_processor.scores
            if scores["left"] and scores["right"]:
                st.session_state.assessments.append({
                    'patient': st.session_state.get('current_patient', 'Unknown'),
                    'exercise': st.session_state.get('exercise', 'Squat'),
                    'left_avg': np.mean(scores["left"]),
                    'right_avg': np.mean(scores["right"]),
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M")
                })
                save_assessments()
                st.success("Assessment Saved!")
    st.info("💡 Stand 6-8 feet away. AI speaks feedback automatically.")

elif page == "Patient Records":
    st.markdown("## Patient Records")
    with st.form("add_patient"):
        c1, c2, c3 = st.columns(3)
        with c1: name = st.text_input("Full Name")
        with c2: age = st.number_input("Age", min_value=1)
        with c3: condition = st.selectbox("Condition", ["Stroke", "Paralysis", "Fracture", "Sports Injury", "Post-Surgery", "Other"])
        village = st.text_input("Village/Location")
        if st.form_submit_button("Register Patient"):
            if name:
                st.session_state.patients.append({
                    'id': len(st.session_state.patients)+1,
                    'name': name, 'age': age, 'condition': condition,
                    'village': village, 'reg_date': datetime.now().strftime("%Y-%m-%d")
                })
                save_patients()
                st.success("Patient Registered")
    if st.session_state.patients:
        st.dataframe(pd.DataFrame(st.session_state.patients), use_container_width=True)

elif page == "Reports":
    st.markdown("## Clinical Reports")
    if st.session_state.assessments:
        df = pd.DataFrame(st.session_state.assessments)
        st.dataframe(df, use_container_width=True)
        fig = px.line(df, x='date', y=['left_avg', 'right_avg'], title='Progress Over Time', markers=True)
        st.plotly_chart(fig, use_container_width=True)
        if st.button("Generate Clinical Report (PDF)"):
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=A4)
            c.setFillColorRGB(0.043, 0.165, 0.294)
            c.rect(0, 800, 600, 80, fill=1)
            c.setFillColorRGB(1,1,1)
            c.setFont("Helvetica-Bold", 16)
            c.drawString(40, 830, "NPRC B.A.D.R v5.0")
            c.setFont("Helvetica", 10)
            c.drawString(40, 810, "Clinical Assessment Report")
            c.setFillColorRGB(0,0,0)
            y = 750
            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, y, "Summary")
            y -= 25
            c.setFont("Helvetica", 10)
            c.drawString(60, y, f"Total Assessments: {len(st.session_state.assessments)}")
            y -= 20
            c.drawString(60, y, f"Average Left Angle: {df['left_avg'].mean():.1f}°")
            y -= 20
            c.drawString(60, y, f"Average Right Angle: {df['right_avg'].mean():.1f}°")
            c.setFillColorRGB(0.043, 0.165, 0.294)
            c.rect(0, 20, 600, 30, fill=1)
            c.setFillColorRGB(1,1,1)
            c.setFont("Helvetica", 8)
            c.drawString(40, 30, "Generated by NPRC Clinical System | Govt. of India")
            c.save()
            buffer.seek(0)
            b64 = base64.b64encode(buffer.getvalue()).decode()
            st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="Clinical_Report.pdf" style="background:#D4AF37; color:#0B2A4A; padding:12px 25px; border-radius:30px; text-decoration:none;">Download PDF</a>', unsafe_allow_html=True)
    else:
        st.info("No assessments recorded yet.")

elif page == "Settings":
    st.markdown("## System Settings")
    st.markdown("---")
    st.markdown("### Data Management")
    if st.button("Export All Data (CSV)"):
        data = {"patients": pd.DataFrame(st.session_state.patients), "assessments": pd.DataFrame(st.session_state.assessments)}
        zip_buffer = io.BytesIO()
        import zipfile
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            for name, df in data.items():
                if not df.empty:
                    zf.writestr(f"{name}.csv", df.to_csv(index=False))
        zip_buffer.seek(0)
        b64 = base64.b64encode(zip_buffer.getvalue()).decode()
        st.markdown(f'<a href="data:application/zip;base64,{b64}" download="NPRC_Data_Backup.zip" style="background:#0B2A4A; color:white; padding:10px 20px; border-radius:30px; text-decoration:none;">Download Backup</a>', unsafe_allow_html=True)

# ---------- फुटर ----------
st.markdown(f"""
<div class="footer">
    <div style="display:flex; justify-content:space-between; flex-wrap:wrap;">
        <div>NPRC Clinical B.A.D.R v5.0 | Govt. of India</div>
        <div>Ministry of Health & Sports Medicine</div>
        <div>Offline System</div>
    </div>
</div>
""", unsafe_allow_html=True)

# पहली बार CSV बनाएं
if __name__ == "__main__":
    for f in ['patients.csv', 'assessments.csv']:
        if not os.path.exists(f):
            pd.DataFrame().to_csv(f, index=False)
